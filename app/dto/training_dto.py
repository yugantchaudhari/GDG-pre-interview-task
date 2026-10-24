"""
app/dto/training_dto.py
========================
Pydantic v2 Data Transfer Objects for the model training endpoint.

All validation bounds and default values are loaded **dynamically** from
``get_settings()`` so that changing a value in ``.env`` automatically
propagates to DTO validation without any code change.
"""

from __future__ import annotations

from app.config.settings import get_settings
from app.dto.resilience_dto import ResilienceMetadataDTO
from pydantic import BaseModel, Field

# Resolve settings once at module import time.  The lru_cache ensures this
# is a single Settings object shared across all imports.
_s = get_settings()


class TrainingRequestDTO(BaseModel):
    """
    Request body for ``POST /api/v1/model/train``.

    All fields have sensible defaults drawn from ``.env`` so that a bare
    ``POST /api/v1/model/train`` with an empty body launches training with
    the configured defaults.
    """

    cv_folds: int = Field(
        default=_s.DEFAULT_CV_FOLDS,
        ge=_s.VALIDATION_CV_FOLDS_MIN,
        le=_s.VALIDATION_CV_FOLDS_MAX,
        description=(
            f"Number of cross-validation folds "
            f"(min={_s.VALIDATION_CV_FOLDS_MIN}, max={_s.VALIDATION_CV_FOLDS_MAX})."
        ),
    )
    drop_duplicates: bool = Field(
        default=_s.DEFAULT_DROP_DUPLICATES,
        description="Whether to drop fully duplicate rows before training.",
    )
    handle_feature_skewness: bool = Field(
        default=_s.DEFAULT_HANDLE_FEATURE_SKEW,
        description=(
            "When True, applies PowerTransformer to numerical features whose "
            "absolute skewness exceeds SKEWNESS_HIGH_THRESHOLD."
        ),
    )
    apply_log_transform_on_target: bool = Field(
        default=_s.DEFAULT_APPLY_LOG_TARGET,
        description=(
            "When True, wraps the pipeline in TransformedTargetRegressor "
            "(log1p / expm1) to handle the heavy right-skew of 'charges'."
        ),
    )
    include_bmi_smoker_interaction: bool = Field(
        default=_s.DEFAULT_INCLUDE_INTERACTIONS,
        description="When True, engineers BMI×smoker interaction and age-obesity features.",
    )
    random_state: int = Field(
        default=_s.DEFAULT_RANDOM_STATE,
        description="Global random seed for reproducibility across CV splits and estimators.",
    )
    dataset_path_override: str | None = Field(
        default=None,
        description=(
            "Optional alternative path to a CSV dataset. "
            "When None, DATASET_FILE_PATH from .env is used."
        ),
    )


class PreprocessingAppliedDTO(BaseModel):
    """Summary of the preprocessing steps applied during the training run."""

    duplicates_removed: int = Field(
        ..., description="Number of duplicate rows dropped from the raw dataset."
    )
    categorical_encoding: str = Field(
        ..., description="Description of the categorical encoding strategy used."
    )
    numerical_scaling: str = Field(
        ..., description="Description of the numerical scaling / transformation strategy."
    )
    target_skew_transform: str = Field(
        ..., description="Description of the target variable transformation (or 'None')."
    )
    engineered_features: list[str] = Field(
        ..., description="List of newly engineered feature column names added to the feature matrix."
    )


class ModelFoldMetricsDTO(BaseModel):
    """Aggregated cross-validation fold metrics for a single trained model."""

    mae_mean: float = Field(..., description="Mean Absolute Error — cross-validation mean.")
    mae_std: float = Field(..., description="Mean Absolute Error — cross-validation standard deviation.")
    rmse_mean: float = Field(..., description="Root Mean Squared Error — cross-validation mean.")
    rmse_std: float = Field(..., description="Root Mean Squared Error — cross-validation standard deviation.")
    r2_mean: float = Field(..., description="R² coefficient of determination — cross-validation mean.")
    r2_std: float = Field(..., description="R² coefficient of determination — cross-validation standard deviation.")


class LeaderboardEntryDTO(BaseModel):
    """A single row in the model evaluation leaderboard."""

    rank: int = Field(..., description="Rank position (1 = best).")
    model_name: str = Field(..., description="Scikit-learn estimator class name.")
    metrics: ModelFoldMetricsDTO = Field(..., description="Aggregated CV fold metrics.")
    is_best: bool = Field(..., description="True for the model selected as the best pipeline.")


class SavedArtifactsDTO(BaseModel):
    """Paths of artefact files persisted after training."""

    best_pipeline_path: str = Field(
        ..., description="Filesystem path to the serialised best pipeline (.joblib)."
    )
    all_models_path: str = Field(
        ..., description="Filesystem path to the dict of all trained pipelines (.joblib)."
    )
    metrics_report_path: str = Field(
        ..., description="Filesystem path to the JSON evaluation report."
    )


class TrainingResponseDTO(BaseModel):
    """
    Response body for ``POST /api/v1/model/train`` and
    ``GET /api/v1/model/metrics``.
    """

    status: str = Field(..., description="Training run status ('SUCCESS' or 'DEGRADED_FALLBACK_NO_MODEL_TRAINED_YET').")
    trained_at: str = Field(..., description="ISO-8601 UTC timestamp of training completion.")
    preprocessing_applied: PreprocessingAppliedDTO = Field(
        ..., description="Summary of all preprocessing steps applied during this training run."
    )
    cv_folds: int = Field(..., description="Number of cross-validation folds used.")
    best_model_selected: str = Field(
        ..., description="Name of the model with the highest mean R² (lowest RMSE on tie)."
    )
    selection_criterion: str = Field(
        ..., description="Human-readable description of the model selection criterion."
    )
    leaderboard: list[LeaderboardEntryDTO] = Field(
        ..., description="Ranked list of all evaluated models with their CV metrics."
    )
    saved_artifacts: SavedArtifactsDTO = Field(
        ..., description="Paths to persisted artefacts written after training."
    )
    resilience_metadata: ResilienceMetadataDTO | None = Field(
        default=None,
        description="Resilience4j execution telemetry injected by the @resilience4j_pipeline decorator.",
    )
