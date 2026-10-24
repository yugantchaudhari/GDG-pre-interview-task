"""
app/dto/prediction_dto.py
==========================
Pydantic v2 Data Transfer Objects for the real-time prediction endpoint.

All field validation bounds (``ge`` / ``le``) are loaded dynamically from
``get_settings()`` so that changing ``.env`` values propagates automatically
to request validation without touching any source code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.dto.resilience_dto import ResilienceMetadataDTO

# Resolve settings once at import time (lru_cache guarantees single object).
_s = get_settings()


class PredictionRequestDTO(BaseModel):
    """
    Request body for ``POST /api/v1/predict``.

    All categorical fields use ``Literal`` types so that FastAPI's
    OpenAPI schema renders proper enum dropdowns in the Swagger UI.
    """

    age: int = Field(
        ...,
        ge=_s.VALIDATION_AGE_MIN,
        le=_s.VALIDATION_AGE_MAX,
        examples=[34],
        description=(
            f"Policyholder's age in years "
            f"(min={_s.VALIDATION_AGE_MIN}, max={_s.VALIDATION_AGE_MAX})."
        ),
    )
    sex: Literal["male", "female"] = Field(
        ...,
        examples=["male"],
        description="Policyholder's biological sex.",
    )
    bmi: float = Field(
        ...,
        ge=_s.VALIDATION_BMI_MIN,
        le=_s.VALIDATION_BMI_MAX,
        examples=[31.4],
        description=(
            f"Body Mass Index "
            f"(min={_s.VALIDATION_BMI_MIN}, max={_s.VALIDATION_BMI_MAX})."
        ),
    )
    children: int = Field(
        ...,
        ge=_s.VALIDATION_CHILDREN_MIN,
        le=_s.VALIDATION_CHILDREN_MAX,
        examples=[2],
        description=(
            f"Number of dependents covered by the plan "
            f"(min={_s.VALIDATION_CHILDREN_MIN}, max={_s.VALIDATION_CHILDREN_MAX})."
        ),
    )
    smoker: Literal["yes", "no"] = Field(
        ...,
        examples=["yes"],
        description="Whether the policyholder is a tobacco smoker.",
    )
    region: Literal["northeast", "northwest", "southeast", "southwest"] = Field(
        ...,
        examples=["southeast"],
        description="US geographic region of the policyholder's residence.",
    )
    model_override: (
        Literal[
            "LinearRegression",
            "RandomForestRegressor",
            "GradientBoostingRegressor",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Optional model name to use for inference. "
            "When None, the best-ranked model is used. "
            "Must match a name from the evaluation leaderboard."
        ),
    )


class InputSummaryDTO(BaseModel):
    """
    Echo of the parsed and enriched prediction request fields.

    Included in the response so clients can confirm what was received and
    verify the BMI classification the service applied.
    """

    age: int = Field(..., description="Policyholder's age.")
    sex: str = Field(..., description="Policyholder's biological sex.")
    bmi: float = Field(..., description="Policyholder's Body Mass Index.")
    bmi_category: str = Field(
        ...,
        description=(
            "BMI classification derived from domain thresholds: "
            "'Underweight', 'Normal', 'Overweight', or 'Obese'."
        ),
    )
    children: int = Field(..., description="Number of dependents.")
    smoker: str = Field(..., description="Smoking status ('yes' / 'no').")
    region: str = Field(..., description="US geographic region.")


class RiskAssessmentDTO(BaseModel):
    """
    Actuarial risk profile derived from the policyholder's input features.

    The ``risk_tier`` and ``key_cost_drivers`` are computed from domain rules
    defined in ``.env`` thresholds — no hardcoding.
    """

    risk_tier: Literal["LOW_RISK", "MODERATE_RISK", "HIGH_RISK"] = Field(
        ...,
        description=(
            "Risk classification: "
            "HIGH_RISK (smoker), "
            "MODERATE_RISK (obese or senior), "
            "LOW_RISK (none of the above)."
        ),
    )
    key_cost_drivers: list[str] = Field(
        ...,
        description=(
            "Ordered list of the primary factors driving predicted cost "
            "(e.g., 'Smoking status', 'Obesity (BMI ≥ 30)', 'Senior age bracket')."
        ),
    )


class PredictionResponseDTO(BaseModel):
    """
    Response body for ``POST /api/v1/predict``.

    Combines the raw predicted charge, a formatted currency string, the model
    used, input echo, risk assessment, and optional Resilience4j telemetry.
    """

    predicted_charge: float = Field(
        ...,
        description="Raw numeric predicted annual insurance charge (in DEFAULT_CURRENCY).",
    )
    currency: str = Field(
        ...,
        description="ISO 4217 currency code (from DEFAULT_CURRENCY in .env).",
    )
    formatted_charge: str = Field(
        ...,
        description="Human-readable charge string with currency symbol and comma formatting.",
    )
    model_used: str = Field(
        ...,
        description=(
            "Name of the scikit-learn estimator (or fallback engine) that "
            "produced this prediction."
        ),
    )
    input_summary: InputSummaryDTO = Field(
        ...,
        description="Echo and enrichment of the validated request fields.",
    )
    risk_assessment: RiskAssessmentDTO = Field(
        ..., description="Actuarial risk tier and identified cost drivers."
    )
    resilience_metadata: ResilienceMetadataDTO | None = Field(
        default=None,
        description="Resilience4j execution telemetry injected by @resilience4j_pipeline.",
    )
