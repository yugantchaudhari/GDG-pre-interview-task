"""
app/services/training_service.py
==================================
TrainingService — orchestrates 5-fold cross-validated model comparison,
artefact persistence, and evaluation report generation.

Workflow
--------
1. Load and clean dataset via ``DatasetService``.
2. Separate features ``X`` and target ``y``.
3. Initialise 3 candidate regressors from ``.env`` hyperparameters.
4. Wrap each in a full pipeline via ``PreprocessingService``.
5. Run ``cross_validate`` with MAE, RMSE, and R² scoring.
6. Rank by ``r2_mean`` (descending), breaking ties by ``rmse_mean`` (ascending).
7. Fit the winning pipeline on full ``(X, y)`` and save all artefacts via ``joblib``.
8. Write the ``TrainingResponseDTO`` as a JSON evaluation report to disk.
9. Hot-reload the newly trained pipeline into ``PredictionService`` memory.

No hardcoded hyperparameters — every value comes from ``get_settings()``.
No ``print()`` statements anywhere.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate

from app.config.settings import get_settings
from app.dto.training_dto import (
    LeaderboardEntryDTO,
    ModelFoldMetricsDTO,
    PreprocessingAppliedDTO,
    SavedArtifactsDTO,
    TrainingRequestDTO,
    TrainingResponseDTO,
)
from app.exceptions.custom_exceptions import ModelNotTrainedException
from app.services.dataset_service import DatasetService
from app.services.preprocessing_service import PreprocessingService
from app.utils.logger import LoggerFactory

if TYPE_CHECKING:
    from app.services.prediction_service import PredictionService

log = LoggerFactory.get_logger(__name__)

_FEATURE_COLUMNS: list[str] = ["age", "sex", "bmi", "children", "smoker", "region"]
_TARGET_COLUMN: str = "charges"


class TrainingService:
    """
    Orchestrates multi-model cross-validated training and artefact persistence.

    Parameters
    ----------
    dataset_service:
        Injected ``DatasetService`` used to load the training data.
    preprocessing_service:
        Injected ``PreprocessingService`` used to build pipeline instances.
    prediction_service:
        Injected ``PredictionService`` whose in-memory state is refreshed
        after a successful training run.
    """

    def __init__(
        self,
        dataset_service: DatasetService,
        preprocessing_service: PreprocessingService,
        prediction_service: "PredictionService",
    ) -> None:
        self.settings = get_settings()
        self._dataset_service = dataset_service
        self._preprocessing_service = preprocessing_service
        self._prediction_service = prediction_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_and_evaluate(
        self, request: TrainingRequestDTO | None = None
    ) -> TrainingResponseDTO:
        """
        Run a full cross-validated training run and return a ranked evaluation report.

        Parameters
        ----------
        request:
            Training configuration.  Uses ``TrainingRequestDTO()`` defaults
            when ``None``.

        Returns
        -------
        TrainingResponseDTO
            Fully populated response with leaderboard, artefact paths, and
            preprocessing summary.
        """
        if request is None:
            request = TrainingRequestDTO()

        log.info(
            "Training run started — cv_folds=%d, random_state=%d, "
            "drop_duplicates=%s, log_target=%s, interactions=%s.",
            request.cv_folds,
            request.random_state,
            request.drop_duplicates,
            request.apply_log_transform_on_target,
            request.include_bmi_smoker_interaction,
        )

        # ----------------------------------------------------------------
        # 1. Load and clean dataset
        # ----------------------------------------------------------------
        df, duplicates_removed = self._dataset_service.get_clean_dataset(
            drop_duplicates=request.drop_duplicates,
            custom_path=request.dataset_path_override,
        )

        X = df[_FEATURE_COLUMNS].copy()
        y = df[_TARGET_COLUMN].copy()

        log.info("Feature matrix shape: %s | Target shape: %s", X.shape, y.shape)

        # ----------------------------------------------------------------
        # 2. Initialise candidate estimators from settings
        # ----------------------------------------------------------------
        s = self.settings
        candidate_estimators: dict[str, Any] = {
            "LinearRegression": LinearRegression(
                fit_intercept=s.LR_FIT_INTERCEPT
            ),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=s.RF_N_ESTIMATORS,
                max_depth=s.RF_MAX_DEPTH,
                min_samples_split=s.RF_MIN_SAMPLES_SPLIT,
                min_samples_leaf=s.RF_MIN_SAMPLES_LEAF,
                n_jobs=s.RF_N_JOBS,
                random_state=request.random_state,
            ),
            "GradientBoostingRegressor": GradientBoostingRegressor(
                n_estimators=s.GB_N_ESTIMATORS,
                learning_rate=s.GB_LEARNING_RATE,
                max_depth=s.GB_MAX_DEPTH,
                subsample=s.GB_SUBSAMPLE,
                min_samples_split=s.GB_MIN_SAMPLES_SPLIT,
                random_state=request.random_state,
            ),
        }

        # ----------------------------------------------------------------
        # 3. Cross-validation configuration
        # ----------------------------------------------------------------
        kfold = KFold(
            n_splits=request.cv_folds,
            shuffle=True,
            random_state=request.random_state,
        )

        # ----------------------------------------------------------------
        # 4. Build, evaluate, and fit each candidate pipeline
        # ----------------------------------------------------------------
        cv_results: dict[str, dict[str, float]] = {}
        fitted_pipelines: dict[str, Any] = {}

        for model_name, estimator in candidate_estimators.items():
            log.info("Evaluating model: %s", model_name)
            pipeline = self._preprocessing_service.build_full_model_pipeline(
                base_estimator=estimator,
                handle_feature_skewness=request.handle_feature_skewness,
                apply_log_target=request.apply_log_transform_on_target,
                include_interactions=request.include_bmi_smoker_interaction,
            )

            cv_scores = cross_validate(
                pipeline,
                X,
                y,
                cv=kfold,
                scoring=(
                    "neg_mean_absolute_error",
                    "neg_root_mean_squared_error",
                    "r2",
                ),
                n_jobs=1,
                return_train_score=False,
            )

            mae_scores = -cv_scores["test_neg_mean_absolute_error"]
            rmse_scores = -cv_scores["test_neg_root_mean_squared_error"]
            r2_scores = cv_scores["test_r2"]

            cv_results[model_name] = {
                "mae_mean": float(np.mean(mae_scores)),
                "mae_std": float(np.std(mae_scores)),
                "rmse_mean": float(np.mean(rmse_scores)),
                "rmse_std": float(np.std(rmse_scores)),
                "r2_mean": float(np.mean(r2_scores)),
                "r2_std": float(np.std(r2_scores)),
            }

            log.info(
                "%s CV results: MAE=%.2f±%.2f | RMSE=%.2f±%.2f | R²=%.4f±%.4f",
                model_name,
                cv_results[model_name]["mae_mean"],
                cv_results[model_name]["mae_std"],
                cv_results[model_name]["rmse_mean"],
                cv_results[model_name]["rmse_std"],
                cv_results[model_name]["r2_mean"],
                cv_results[model_name]["r2_std"],
            )

            # Fit on full dataset for inference readiness
            pipeline.fit(X, y)
            fitted_pipelines[model_name] = pipeline

        # ----------------------------------------------------------------
        # 5. Rank models (highest R² → lowest RMSE as tie-breaker)
        # ----------------------------------------------------------------
        ranked = sorted(
            cv_results.items(),
            key=lambda kv: (-kv[1]["r2_mean"], kv[1]["rmse_mean"]),
        )
        best_model_name = ranked[0][0]
        log.info(
            "Model ranking complete. Best model: %s (R²=%.4f, RMSE=%.2f).",
            best_model_name,
            cv_results[best_model_name]["r2_mean"],
            cv_results[best_model_name]["rmse_mean"],
        )

        # ----------------------------------------------------------------
        # 6. Build leaderboard
        # ----------------------------------------------------------------
        leaderboard: list[LeaderboardEntryDTO] = []
        for rank, (model_name, metrics) in enumerate(ranked, start=1):
            leaderboard.append(
                LeaderboardEntryDTO(
                    rank=rank,
                    model_name=model_name,
                    metrics=ModelFoldMetricsDTO(**metrics),
                    is_best=(model_name == best_model_name),
                )
            )

        # ----------------------------------------------------------------
        # 7. Save artefacts
        # ----------------------------------------------------------------
        best_pipeline = fitted_pipelines[best_model_name]
        joblib.dump(best_pipeline, self.settings.best_pipeline_path)
        log.info("Best pipeline saved to '%s'.", self.settings.best_pipeline_path)

        all_models_payload: dict[str, Any] = {
            "pipelines": fitted_pipelines,
            "best_model_name": best_model_name,
            "cv_results": cv_results,
        }
        joblib.dump(all_models_payload, self.settings.all_models_path)
        log.info("All pipelines saved to '%s'.", self.settings.all_models_path)

        # ----------------------------------------------------------------
        # 8. Assemble response DTO
        # ----------------------------------------------------------------
        trained_at = datetime.now(timezone.utc).isoformat()

        engineered = (
            ["is_obese", "bmi_smoker_interaction", "age_bmi_interaction"]
            if request.include_bmi_smoker_interaction
            else []
        )
        preprocessing_desc = PreprocessingAppliedDTO(
            duplicates_removed=duplicates_removed,
            categorical_encoding=(
                f"OneHotEncoder(binary drop={s.OHE_DROP_STRATEGY}) + "
                f"OneHotEncoder(nominal handle_unknown={s.OHE_HANDLE_UNKNOWN})"
            ),
            numerical_scaling=(
                f"PowerTransformer({s.POWER_TRANSFORM_METHOD}) + StandardScaler"
                if request.handle_feature_skewness
                else "StandardScaler"
            ),
            target_skew_transform=(
                "TransformedTargetRegressor(log1p / expm1)"
                if request.apply_log_transform_on_target
                else "None"
            ),
            engineered_features=engineered,
        )

        saved_artifacts = SavedArtifactsDTO(
            best_pipeline_path=str(self.settings.best_pipeline_path),
            all_models_path=str(self.settings.all_models_path),
            metrics_report_path=str(self.settings.evaluation_report_path),
        )

        response = TrainingResponseDTO(
            status="SUCCESS",
            trained_at=trained_at,
            preprocessing_applied=preprocessing_desc,
            cv_folds=request.cv_folds,
            best_model_selected=best_model_name,
            selection_criterion="Highest mean R² across CV folds (lowest RMSE on tie)",
            leaderboard=leaderboard,
            saved_artifacts=saved_artifacts,
        )

        # ----------------------------------------------------------------
        # 9. Write JSON evaluation report
        # ----------------------------------------------------------------
        report_dict = response.model_dump()
        self.settings.evaluation_report_path.write_text(
            json.dumps(report_dict, indent=2, default=str),
            encoding="utf-8",
        )
        log.info(
            "Evaluation report written to '%s'.", self.settings.evaluation_report_path
        )

        # ----------------------------------------------------------------
        # 10. Hot-reload artefacts into PredictionService
        # ----------------------------------------------------------------
        self._prediction_service.load_artifacts_into_memory()

        log.info("Training run completed successfully. Best model: %s.", best_model_name)
        return response

    def get_saved_metrics(self) -> TrainingResponseDTO:
        """
        Load and return the persisted evaluation report from disk.

        Returns
        -------
        TrainingResponseDTO
            The last successful training report deserialized from JSON.

        Raises
        ------
        ModelNotTrainedException
            When no evaluation report exists on disk (model not yet trained).
        """
        report_path: Path = self.settings.evaluation_report_path
        if not report_path.exists():
            log.warning(
                "Evaluation report not found at '%s' — model not yet trained.",
                report_path,
            )
            raise ModelNotTrainedException(
                details={"evaluation_report_path": str(report_path)}
            )

        raw = json.loads(report_path.read_text(encoding="utf-8"))
        log.info("Loaded saved evaluation report from '%s'.", report_path)
        return TrainingResponseDTO.model_validate(raw)
