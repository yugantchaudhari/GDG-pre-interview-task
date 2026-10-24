"""
app/services/eda_service.py
============================
EdaService — Exploratory Data Analysis for the insurance dataset.

This service computes dataset-level statistics (row/column counts, duplicates,
missing values), per-feature descriptive statistics with skewness flags, value
frequencies for categorical columns, and actuarial domain insights
(smoker cost multiplier, Pearson correlations).

All domain thresholds (``SKEWNESS_HIGH_THRESHOLD``) are read from ``Settings``
— no hardcoded values anywhere.  No ``print()`` calls used.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.config.settings import get_settings
from app.dto.eda_dto import (
    DomainInsightsDTO,
    EdaSummaryResponseDTO,
    NumericalFeatureStatsDTO,
)
from app.services.dataset_service import DatasetService
from app.utils.logger import LoggerFactory

log = LoggerFactory.get_logger(__name__)


def _to_float(val: Any) -> float:
    """Safely converts Pandas/NumPy scalar objects to a native Python float."""
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)

_NUMERICAL_FEATURES: list[str] = ["age", "bmi", "children", "charges"]
_CATEGORICAL_FEATURES: list[str] = ["sex", "smoker", "region"]
_CORRELATION_FEATURES: list[str] = ["age", "bmi", "children"]


class EdaService:
    """
    Exploratory Data Analysis service.

    Loads the raw dataset and computes a rich statistical summary without
    mutating the stored data.  The result is a fully-typed ``EdaSummaryResponseDTO``.
    """

    def __init__(self, dataset_service: DatasetService) -> None:
        self.settings = get_settings()
        self._dataset_service = dataset_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_eda_summary(
        self, custom_path: str | None = None
    ) -> EdaSummaryResponseDTO:
        """
        Compute and return a full EDA summary for the insurance dataset.

        Parameters
        ----------
        custom_path:
            Optional override path to the dataset CSV.

        Returns
        -------
        EdaSummaryResponseDTO
            Fully populated response DTO.
        """
        df = self._dataset_service.load_raw_dataset(custom_path)

        total_rows = len(df)
        total_columns = len(df.columns)
        duplicate_rows = int(df.duplicated().sum())
        missing_values: dict[str, int] = {
            col: df[col].isna().sum() for col in df.columns
        }

        log.info(
            "EDA summary: %d rows, %d columns, %d duplicates, %d total missing values.",
            total_rows,
            total_columns,
            duplicate_rows,
            sum(missing_values.values()),
        )

        numerical_summary = self._compute_numerical_summary(df)
        categorical_counts = self._compute_categorical_counts(df)
        domain_insights = self._compute_domain_insights(df)

        return EdaSummaryResponseDTO(
            total_rows=total_rows,
            total_columns=total_columns,
            duplicate_rows=duplicate_rows,
            missing_values=missing_values,
            numerical_summary=numerical_summary,
            categorical_counts=categorical_counts,
            domain_insights=domain_insights,
        )

    # ------------------------------------------------------------------
    # Private computation helpers
    # ------------------------------------------------------------------

    def _compute_numerical_summary(
        self, df: pd.DataFrame
    ) -> dict[str, NumericalFeatureStatsDTO]:
        """Compute descriptive stats for each numerical feature."""
        summary: dict[str, NumericalFeatureStatsDTO] = {}
        for col in _NUMERICAL_FEATURES:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            skew_val = _to_float(series.skew())
            summary[col] = NumericalFeatureStatsDTO(
                mean=round(_to_float(series.mean()), 4),
                std=round(_to_float(series.std()), 4),
                min=round(_to_float(series.min()), 4),
                max=round(_to_float(series.max()), 4),
                skewness=round(skew_val, 4),
                is_highly_skewed=abs(skew_val) >= self.settings.SKEWNESS_HIGH_THRESHOLD,
            )
            log.debug(
                "EDA [%s]: mean=%.4f std=%.4f skew=%.4f highly_skewed=%s",
                col,
                summary[col].mean,
                summary[col].std,
                summary[col].skewness,
                summary[col].is_highly_skewed,
            )
        return summary

    def _compute_categorical_counts(
        self, df: pd.DataFrame
    ) -> dict[str, dict[str, int]]:
        """Compute value_counts() for each categorical feature."""
        counts: dict[str, dict[str, int]] = {}
        for col in _CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
            counts[col] = {
                str(k): v for k, v in df[col].value_counts().items()
            }
            log.debug("EDA [%s] value counts: %s", col, counts[col])
        return counts

    def _compute_domain_insights(self, df: pd.DataFrame) -> DomainInsightsDTO:
        """Compute actuarial domain metrics: smoker premium and correlations."""
        smokers = df[df["smoker"] == "yes"]["charges"]
        non_smokers = df[df["smoker"] == "no"]["charges"]

        avg_smoker = _to_float(smokers.mean()) if not smokers.empty else 0.0
        avg_non_smoker = _to_float(non_smokers.mean()) if not non_smokers.empty else 0.0
        multiplier = (
            round(avg_smoker / avg_non_smoker, 4) if avg_non_smoker > 0 else 0.0
        )

        correlation_with_charges: dict[str, float] = {}
        for col in _CORRELATION_FEATURES:
            if col in df.columns and "charges" in df.columns:
                corr = _to_float(df[[col, "charges"]].corr().loc[col, "charges"])
                correlation_with_charges[col] = round(corr, 4)

        log.info(
            "Domain insights: avg_smoker=%.2f avg_non_smoker=%.2f multiplier=%.2fx",
            avg_smoker,
            avg_non_smoker,
            multiplier,
        )

        return DomainInsightsDTO(
            avg_charge_smoker=round(avg_smoker, 2),
            avg_charge_non_smoker=round(avg_non_smoker, 2),
            smoker_cost_multiplier=multiplier,
            correlation_with_charges=correlation_with_charges,
        )
