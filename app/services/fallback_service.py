"""
app/services/fallback_service.py
==================================
FallbackService — the Resilience4j ``fallbackMethod`` provider.

This service computes actuarially-derived insurance charge estimates using
formula weights defined entirely in ``.env`` (Groups 6 & 3).  It is invoked
by the ``@resilience4j_pipeline`` decorator when the primary prediction or
metrics path raises an exception and the relevant fallback toggle is enabled.

Actuarial formula
-----------------
    estimated = base
              + (age × age_multiplier)
              + max(0, bmi - bmi_baseline) × excess_bmi_multiplier
              + children × children_multiplier
              + smoker_base_penalty       (if smoker == "yes")
              + smoker_obese_synergy      (if smoker == "yes" AND bmi >= BMI_OBESE_THRESHOLD)

No hardcoded constants.  No ``print()`` calls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.config.settings import get_settings
from app.dto.prediction_dto import (
    InputSummaryDTO,
    PredictionRequestDTO,
    PredictionResponseDTO,
    RiskAssessmentDTO,
)
from app.dto.resilience_dto import ResilienceMetadataDTO
from app.dto.training_dto import (
    LeaderboardEntryDTO,
    ModelFoldMetricsDTO,
    PreprocessingAppliedDTO,
    SavedArtifactsDTO,
    TrainingResponseDTO,
)
from app.utils.logger import LoggerFactory

log = LoggerFactory.get_logger(__name__)

_FALLBACK_MODEL_NAME = "ActuarialHeuristicFallbackEngine"


class FallbackService:
    """
    Provides fallback responses for the ``predict`` and ``metrics`` endpoints
    when the primary Resilience4j-guarded path is unavailable.

    All actuarial formula weights are read from ``get_settings()`` so that
    the financial model can be tuned via ``.env`` without code changes.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_charge_fallback(
        self,
        request: PredictionRequestDTO,
        reason: str,
        resilience_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> PredictionResponseDTO:
        """
        Compute an actuarially-estimated insurance charge as a fallback.

        This method is called by the ``@resilience4j_pipeline`` decorator when
        the ML-powered ``predict_charge`` raises any exception and
        ``ENABLE_FALLBACK_ON_PREDICT=True``.

        Parameters
        ----------
        request:
            The original validated prediction request.
        reason:
            String representation of the exception that triggered the fallback.
        resilience_metadata:
            Mutable metadata dict built by the decorator (already contains
            ``fallback_triggered=True`` at this point).

        Returns
        -------
        PredictionResponseDTO
            A degraded but valid response with model_used indicating the
            fallback engine.
        """
        s = self.settings

        log.warning(
            "FallbackService.predict_charge_fallback activated — reason: %s | "
            "age=%d bmi=%.1f smoker=%s.",
            reason,
            request.age,
            request.bmi,
            request.smoker,
        )

        # ----------------------------------------------------------------
        # Actuarial formula — all weights from .env
        # ----------------------------------------------------------------
        base = s.FALLBACK_BASE_CHARGE
        age_component = request.age * s.FALLBACK_AGE_MULTIPLIER
        excess_bmi = max(0.0, request.bmi - s.FALLBACK_BMI_BASELINE)
        bmi_component = excess_bmi * s.FALLBACK_EXCESS_BMI_MULTIPLIER
        children_component = request.children * s.FALLBACK_CHILDREN_MULTIPLIER
        smoker_penalty = (
            s.FALLBACK_SMOKER_BASE_PENALTY if request.smoker == "yes" else 0.0
        )
        synergy_penalty = (
            s.FALLBACK_SMOKER_OBESE_SYNERGY_PENALTY
            if (request.smoker == "yes" and request.bmi >= s.BMI_OBESE_THRESHOLD)
            else 0.0
        )

        estimated = (
            base
            + age_component
            + bmi_component
            + children_component
            + smoker_penalty
            + synergy_penalty
        )
        estimated = max(0.0, estimated)

        log.info(
            "Actuarial estimate: base=%.2f age_comp=%.2f bmi_comp=%.2f "
            "children_comp=%.2f smoker_penalty=%.2f synergy=%.2f → total=%.2f",
            base,
            age_component,
            bmi_component,
            children_component,
            smoker_penalty,
            synergy_penalty,
            estimated,
        )

        # ----------------------------------------------------------------
        # Build resilience metadata DTO
        # ----------------------------------------------------------------
        if resilience_metadata is not None:
            meta_dto = ResilienceMetadataDTO(
                circuit_breaker_name=resilience_metadata.get(
                    "circuit_breaker_name", "predict-cb"
                ),
                circuit_breaker_state=resilience_metadata.get(
                    "circuit_breaker_state", "UNKNOWN"
                ),
                fallback_triggered=True,
                fallback_reason=reason,
                retry_attempts_used=resilience_metadata.get("retry_attempts_used", 0),
                bulkhead_available_permits=resilience_metadata.get(
                    "bulkhead_available_permits", 0
                ),
                rate_limit_remaining=resilience_metadata.get("rate_limit_remaining", 0),
                execution_time_ms=resilience_metadata.get("execution_time_ms", 0.0),
            )
        else:
            meta_dto = None

        # ----------------------------------------------------------------
        # BMI category & risk assessment (same logic as PredictionService)
        # ----------------------------------------------------------------
        if request.bmi < s.BMI_UNDERWEIGHT_THRESHOLD:
            bmi_category = "Underweight"
        elif request.bmi < s.BMI_NORMAL_THRESHOLD:
            bmi_category = "Normal"
        elif request.bmi < s.BMI_OBESE_THRESHOLD:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"

        is_smoker = request.smoker == "yes"
        is_obese = request.bmi >= s.BMI_OBESE_THRESHOLD
        is_senior = request.age >= s.AGE_SENIOR_RISK_THRESHOLD

        drivers: list[str] = []
        if is_smoker:
            drivers.append("Smoking status")
        if is_obese:
            drivers.append(f"Obesity (BMI ≥ {s.BMI_OBESE_THRESHOLD})")
        if is_senior:
            drivers.append(f"Senior age bracket (age ≥ {s.AGE_SENIOR_RISK_THRESHOLD})")
        if request.children > 0:
            drivers.append(f"Number of dependents ({request.children})")
        if not drivers:
            drivers.append("No significant individual risk factors identified")

        if is_smoker:
            risk_tier = "HIGH_RISK"
        elif is_obese or is_senior:
            risk_tier = "MODERATE_RISK"
        else:
            risk_tier = "LOW_RISK"

        return PredictionResponseDTO(
            predicted_charge=round(estimated, 2),
            currency=s.DEFAULT_CURRENCY,
            formatted_charge=f"${estimated:,.2f} (Estimated via Fallback)",
            model_used=_FALLBACK_MODEL_NAME,
            input_summary=InputSummaryDTO(
                age=request.age,
                sex=request.sex,
                bmi=request.bmi,
                bmi_category=bmi_category,
                children=request.children,
                smoker=request.smoker,
                region=request.region,
            ),
            risk_assessment=RiskAssessmentDTO(
                risk_tier=risk_tier,
                key_cost_drivers=drivers,
            ),
            resilience_metadata=meta_dto,
        )

    def get_metrics_fallback(
        self,
        reason: str,
        resilience_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TrainingResponseDTO:
        """
        Return a safe degraded ``TrainingResponseDTO`` when the primary metrics
        path is unavailable (circuit breaker open, model not trained, etc.).

        Parameters
        ----------
        reason:
            String representation of the exception that triggered the fallback.
        resilience_metadata:
            Mutable metadata dict built by the decorator.

        Returns
        -------
        TrainingResponseDTO
            A minimally populated response with status ``DEGRADED_FALLBACK_NO_MODEL_TRAINED_YET``
            and an empty leaderboard.
        """
        log.warning(
            "FallbackService.get_metrics_fallback activated — reason: %s.", reason
        )

        s = self.settings
        now_iso = datetime.now(timezone.utc).isoformat()

        fallback_meta: ResilienceMetadataDTO | None = None
        if resilience_metadata is not None:
            fallback_meta = ResilienceMetadataDTO(
                circuit_breaker_name=resilience_metadata.get(
                    "circuit_breaker_name", "train-cb"
                ),
                circuit_breaker_state=resilience_metadata.get(
                    "circuit_breaker_state", "UNKNOWN"
                ),
                fallback_triggered=True,
                fallback_reason=reason,
                retry_attempts_used=resilience_metadata.get("retry_attempts_used", 0),
                bulkhead_available_permits=resilience_metadata.get(
                    "bulkhead_available_permits", 0
                ),
                rate_limit_remaining=resilience_metadata.get("rate_limit_remaining", 0),
                execution_time_ms=resilience_metadata.get("execution_time_ms", 0.0),
            )

        return TrainingResponseDTO(
            status="DEGRADED_FALLBACK_NO_MODEL_TRAINED_YET",
            trained_at=now_iso,
            preprocessing_applied=PreprocessingAppliedDTO(
                duplicates_removed=0,
                categorical_encoding="N/A (fallback response)",
                numerical_scaling="N/A (fallback response)",
                target_skew_transform="N/A (fallback response)",
                engineered_features=[],
            ),
            cv_folds=s.DEFAULT_CV_FOLDS,
            best_model_selected="N/A",
            selection_criterion="N/A (no model trained yet — fallback response)",
            leaderboard=[],
            saved_artifacts=SavedArtifactsDTO(
                best_pipeline_path=str(s.best_pipeline_path),
                all_models_path=str(s.all_models_path),
                metrics_report_path=str(s.evaluation_report_path),
            ),
            resilience_metadata=fallback_meta,
        )
