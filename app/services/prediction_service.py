"""
app/services/prediction_service.py
=====================================
PredictionService — loads trained artefacts and performs real-time inference.

The service caches the best pipeline and all model pipelines in memory to
eliminate repeated disk I/O during high-frequency prediction calls.

Risk classification and BMI categorisation are driven exclusively by
``Settings`` thresholds (``BMI_OBESE_THRESHOLD``, ``BMI_NORMAL_THRESHOLD``,
``BMI_UNDERWEIGHT_THRESHOLD``, ``AGE_SENIOR_RISK_THRESHOLD``) — no
hardcoded constants anywhere.

No ``print()`` statements are used.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd

from app.config.settings import get_settings
from app.dto.prediction_dto import (
    InputSummaryDTO,
    PredictionRequestDTO,
    PredictionResponseDTO,
    RiskAssessmentDTO,
)
from app.exceptions.custom_exceptions import (
    InvalidModelOverrideException,
    ModelNotTrainedException,
)
from app.utils.logger import LoggerFactory

log = LoggerFactory.get_logger(__name__)

_FEATURE_COLUMNS: list[str] = ["age", "sex", "bmi", "children", "smoker", "region"]


class PredictionService:
    """
    In-memory inference service for the trained insurance pipeline.

    Artefacts are loaded lazily on first prediction call and can be hot-reloaded
    after a new training run without restarting the process.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._best_pipeline: Any | None = None
        self._all_models_dict: dict[str, Any] | None = None
        self._active_best_model_name: str | None = None
        self._last_trained_at: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_artifacts_into_memory(self) -> bool:
        """
        Load the trained pipeline artefacts from disk into instance memory.

        Called automatically on the first prediction request and explicitly
        by ``TrainingService`` after a successful training run (hot-reload).

        Returns
        -------
        bool
            ``True`` if artefacts were loaded successfully; ``False`` if the
            required files do not exist (model not yet trained).
        """
        best_path = self.settings.best_pipeline_path
        all_models_path = self.settings.all_models_path
        report_path = self.settings.evaluation_report_path

        if not best_path.exists() or not all_models_path.exists():
            log.warning(
                "Artefacts not found — best_pipeline='%s' exists=%s, "
                "all_models='%s' exists=%s.",
                best_path,
                best_path.exists(),
                all_models_path,
                all_models_path.exists(),
            )
            return False

        self._best_pipeline = joblib.load(best_path)
        all_models_payload: dict[str, Any] = joblib.load(all_models_path)
        self._all_models_dict = all_models_payload.get("pipelines", {})
        self._active_best_model_name = all_models_payload.get(
            "best_model_name", "Unknown"
        )

        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self._last_trained_at = report.get("trained_at")

        log.info(
            "Artefacts loaded into memory — best_model='%s', "
            "all_models=%s, last_trained_at='%s'.",
            self._active_best_model_name,
            list(self._all_models_dict.keys()),
            self._last_trained_at,
        )
        return True

    def predict_charge(
        self, request: PredictionRequestDTO
    ) -> PredictionResponseDTO:
        """
        Predict the annual insurance charge for a single policyholder.

        Parameters
        ----------
        request:
            Validated ``PredictionRequestDTO`` from the HTTP endpoint.

        Returns
        -------
        PredictionResponseDTO
            The predicted charge with enriched input summary, risk assessment,
            and model provenance information.

        Raises
        ------
        ModelNotTrainedException
            When no trained pipeline is available even after a load attempt.
        InvalidModelOverrideException
            When ``request.model_override`` names a model that was not trained.
        """
        # Lazy load on first call
        if self._best_pipeline is None:
            loaded = self.load_artifacts_into_memory()
            if not loaded or self._best_pipeline is None:
                log.error(
                    "Prediction requested but no trained pipeline is available."
                )
                raise ModelNotTrainedException()

        # Select pipeline (override vs. best)
        if request.model_override is not None:
            if (
                self._all_models_dict is None
                or request.model_override not in self._all_models_dict
            ):
                available = (
                    list(self._all_models_dict.keys())
                    if self._all_models_dict
                    else []
                )
                log.error(
                    "Model override '%s' not found. Available: %s",
                    request.model_override,
                    available,
                )
                raise InvalidModelOverrideException(
                    details={
                        "requested_model": request.model_override,
                        "available_models": available,
                    }
                )
            pipeline = self._all_models_dict[request.model_override]
            model_used = request.model_override
        else:
            pipeline = self._best_pipeline
            model_used = self._active_best_model_name or "BestModel"

        # Build 1-row feature DataFrame
        input_df = pd.DataFrame(
            [
                {
                    "age": request.age,
                    "sex": request.sex,
                    "bmi": request.bmi,
                    "children": request.children,
                    "smoker": request.smoker,
                    "region": request.region,
                }
            ]
        )

        raw_prediction = float(pipeline.predict(input_df)[0])
        # Clip to a realistic floor (never negative charge)
        predicted_charge = max(0.0, raw_prediction)

        log.info(
            "Prediction completed — model='%s' predicted=%.2f for "
            "age=%d bmi=%.1f smoker=%s.",
            model_used,
            predicted_charge,
            request.age,
            request.bmi,
            request.smoker,
        )

        # Build response components
        input_summary = self._build_input_summary(request)
        risk_assessment = self._build_risk_assessment(request)

        s = self.settings
        formatted_charge = f"${predicted_charge:,.2f}"

        return PredictionResponseDTO(
            predicted_charge=round(predicted_charge, 2),
            currency=s.DEFAULT_CURRENCY,
            formatted_charge=formatted_charge,
            model_used=model_used,
            input_summary=input_summary,
            risk_assessment=risk_assessment,
        )

    # ------------------------------------------------------------------
    # Property accessors (used by health controller)
    # ------------------------------------------------------------------

    @property
    def model_trained(self) -> bool:
        """True if artefact files exist on disk."""
        return (
            self.settings.best_pipeline_path.exists()
            and self.settings.all_models_path.exists()
        )

    @property
    def active_best_model_name(self) -> str | None:
        """Name of the currently loaded best model (None if not loaded)."""
        return self._active_best_model_name

    @property
    def last_trained_at(self) -> str | None:
        """ISO-8601 timestamp of the last successful training run."""
        return self._last_trained_at

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_input_summary(
        self, request: PredictionRequestDTO
    ) -> InputSummaryDTO:
        """Build the enriched input echo with BMI category."""
        s = self.settings
        bmi = request.bmi
        if bmi < s.BMI_UNDERWEIGHT_THRESHOLD:
            bmi_category = "Underweight"
        elif bmi < s.BMI_NORMAL_THRESHOLD:
            bmi_category = "Normal"
        elif bmi < s.BMI_OBESE_THRESHOLD:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"

        return InputSummaryDTO(
            age=request.age,
            sex=request.sex,
            bmi=request.bmi,
            bmi_category=bmi_category,
            children=request.children,
            smoker=request.smoker,
            region=request.region,
        )

    def _build_risk_assessment(
        self, request: PredictionRequestDTO
    ) -> RiskAssessmentDTO:
        """
        Compute risk tier and key cost drivers from domain rules in settings.

        Risk tiers:
        - ``HIGH_RISK``: Smoker status (dominant cost factor).
        - ``MODERATE_RISK``: Obese (BMI >= BMI_OBESE_THRESHOLD) OR senior age.
        - ``LOW_RISK``: Neither of the above.
        """
        s = self.settings
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

        return RiskAssessmentDTO(
            risk_tier=risk_tier,
            key_cost_drivers=drivers,
        )
