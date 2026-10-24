"""
app/dto/eda_dto.py
===================
Pydantic v2 Data Transfer Objects for the Exploratory Data Analysis endpoint.

The ``EdaSummaryResponseDTO`` is the top-level response for
``GET /api/v1/eda/summary`` and embeds nested DTOs for numerical feature
statistics, categorical distributions, and actuarial domain insights.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.dto.resilience_dto import ResilienceMetadataDTO


class NumericalFeatureStatsDTO(BaseModel):
    """Descriptive statistics for a single numerical feature."""

    mean: float = Field(..., description="Arithmetic mean of the feature values.")
    std: float = Field(..., description="Standard deviation of the feature values.")
    min: float = Field(..., description="Minimum observed value.")
    max: float = Field(..., description="Maximum observed value.")
    skewness: float = Field(
        ...,
        description=(
            "Fisher-Pearson skewness coefficient. Positive values indicate a "
            "right-skewed distribution; negative values indicate left-skewed."
        ),
    )
    is_highly_skewed: bool = Field(
        ...,
        description=(
            "True when |skewness| >= SKEWNESS_HIGH_THRESHOLD (from .env). "
            "Signals that a power-transform should be considered."
        ),
    )


class DomainInsightsDTO(BaseModel):
    """
    Actuarial domain insights derived from the insurance dataset.

    These metrics quantify the cost impact of smoking and the linear
    associations between numerical risk factors and insurance charges.
    """

    avg_charge_smoker: float = Field(
        ..., description="Mean insurance charge for policyholders who smoke."
    )
    avg_charge_non_smoker: float = Field(
        ..., description="Mean insurance charge for non-smoking policyholders."
    )
    smoker_cost_multiplier: float = Field(
        ...,
        description=(
            "Ratio of avg_charge_smoker to avg_charge_non_smoker. "
            "Indicates how many times more expensive coverage is for smokers."
        ),
    )
    correlation_with_charges: dict[str, float] = Field(
        ...,
        description=(
            "Pearson correlation coefficients between each numerical predictor "
            "(age, bmi, children) and the target variable (charges)."
        ),
    )


class EdaSummaryResponseDTO(BaseModel):
    """
    Top-level response body for the EDA summary endpoint.

    Combines dataset-level statistics, per-feature summaries, categorical
    distributions, and actuarial domain insights into a single structured
    payload.
    """

    total_rows: int = Field(..., description="Total number of rows in the raw dataset.")
    total_columns: int = Field(
        ..., description="Total number of columns in the raw dataset."
    )
    duplicate_rows: int = Field(
        ..., description="Number of fully duplicate rows detected in the raw dataset."
    )
    missing_values: dict[str, int] = Field(
        ...,
        description=(
            "Per-column count of missing (NaN) values. "
            "Keys are column names; values are null-count integers."
        ),
    )
    numerical_summary: dict[str, NumericalFeatureStatsDTO] = Field(
        ...,
        description=(
            "Descriptive statistics for each numerical feature "
            "(age, bmi, children, charges)."
        ),
    )
    categorical_counts: dict[str, dict[str, int]] = Field(
        ...,
        description=(
            "Value-frequency counts for each categorical feature "
            "(sex, smoker, region). Outer key = column name; "
            "inner key = category label; value = row count."
        ),
    )
    domain_insights: DomainInsightsDTO = Field(
        ..., description="Actuarial insights derived from the dataset."
    )
    resilience_metadata: ResilienceMetadataDTO | None = Field(
        default=None,
        description=(
            "Resilience4j execution telemetry injected by the "
            "@resilience4j_pipeline decorator."
        ),
    )
