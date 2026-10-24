"""
app/services/preprocessing_service.py
=======================================
PreprocessingService — custom scikit-learn feature engineering transformer and
full end-to-end model pipeline builder.

Design decisions
----------------
* ``InsuranceFeatureEngineer`` inherits from ``BaseEstimator`` and
  ``TransformerMixin`` at the module top-level so that ``joblib.dump()`` and
  ``joblib.load()`` can serialise and deserialise the fitted pipeline without
  pickling errors.  All constructor arguments must be plain Python types
  (no lambda or closure capturing) so that joblib pickle works correctly.

* ``PreprocessingService.build_full_model_pipeline()`` returns either a
  scikit-learn ``Pipeline`` (when ``apply_log_target=False``) or a
  ``TransformedTargetRegressor`` (when ``apply_log_target=True``).  The
  ``TransformedTargetRegressor`` wraps the full pipeline and applies
  ``np.log1p`` / ``np.expm1`` to the target variable, handling the heavy
  right-skewness of insurance ``charges`` without leaking transformation
  parameters into the feature matrix.

* The ``ColumnTransformer`` uses named sub-pipelines so that feature names
  round-trip correctly through ``get_feature_names_out()``.

No hardcoded values.  All thresholds and strategy strings come from
``get_settings()``.  No ``print()`` calls anywhere.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
)

# TransformedTargetRegressor is imported here; it is only used when
# apply_log_target=True so no circular dependency issues arise.
from sklearn.compose import TransformedTargetRegressor

from app.config.settings import get_settings
from app.utils.logger import LoggerFactory

log = LoggerFactory.get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom scikit-learn transformer — MUST be at module top-level for joblib
# ---------------------------------------------------------------------------

class InsuranceFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer for the insurance dataset.

    Creates interaction features that capture the well-known actuarial synergy
    between obesity and smoking status, and between senior age and obesity.

    Parameters
    ----------
    bmi_obese_threshold:
        BMI value at or above which a policyholder is classified as obese
        (mapped from ``BMI_OBESE_THRESHOLD`` in ``.env``).
    age_senior_threshold:
        Age at or above which a policyholder is in the senior risk bracket
        (mapped from ``AGE_SENIOR_RISK_THRESHOLD`` in ``.env``).
    include_interactions:
        When ``True``, all four interaction features are generated.
        When ``False``, the transformer is a no-op identity pass-through.
    """

    def __init__(
        self,
        bmi_obese_threshold: float = 30.0,
        age_senior_threshold: int = 50,
        include_interactions: bool = True,
    ) -> None:
        self.bmi_obese_threshold = bmi_obese_threshold
        self.age_senior_threshold = age_senior_threshold
        self.include_interactions = include_interactions

    def fit(self, X: Any, y: Any = None) -> "InsuranceFeatureEngineer":  # noqa: N803
        """No-op fit — returns self (stateless transformer)."""
        return self

    def transform(self, X: Any, y: Any = None) -> pd.DataFrame:  # noqa: N803
        """
        Engineer interaction features from the input feature matrix.

        Parameters
        ----------
        X:
            Input DataFrame containing at minimum ``"bmi"``, ``"smoker"``, and
            ``"age"`` columns.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with engineered features appended.
        """
        # Work on a copy to avoid mutating the caller's DataFrame.
        df = pd.DataFrame(X).copy()

        if not self.include_interactions:
            log.debug("InsuranceFeatureEngineer: include_interactions=False — skipping.")
            return df

        # 1. Obesity indicator
        df["is_obese"] = (df["bmi"] >= self.bmi_obese_threshold).astype(int)

        # 2. Numeric smoker indicator (temporary — dropped before return)
        df["is_smoker_numeric"] = (df["smoker"] == "yes").astype(int)

        # 3. BMI × smoker interaction (key actuarial synergy feature)
        df["bmi_smoker_interaction"] = df["bmi"] * df["is_smoker_numeric"]

        # 4. Senior age × obesity cross-risk indicator
        df["age_bmi_interaction"] = (
            (df["age"] >= self.age_senior_threshold).astype(int) * df["is_obese"]
        )

        # Drop the temporary binary smoker column — it would duplicate the
        # downstream OHE-encoded "smoker" column inside the ColumnTransformer.
        df = df.drop(columns=["is_smoker_numeric"])

        log.debug(
            "InsuranceFeatureEngineer: engineered features added — "
            "is_obese, bmi_smoker_interaction, age_bmi_interaction."
        )
        return df

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        """Return feature names after transformation (for pipeline introspection)."""
        base = list(input_features) if input_features is not None else []
        if self.include_interactions:
            base += ["is_obese", "bmi_smoker_interaction", "age_bmi_interaction"]
        return base


# ---------------------------------------------------------------------------
# Preprocessing service
# ---------------------------------------------------------------------------

class PreprocessingService:
    """
    Builds complete scikit-learn preprocessing + regression pipelines.

    All OHE strategies and PowerTransformer methods are read from
    ``Settings`` so changing ``.env`` immediately affects the next training
    run without touching this code.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_full_model_pipeline(
        self,
        base_estimator: Any,
        handle_feature_skewness: bool,
        apply_log_target: bool,
        include_interactions: bool,
    ) -> Any:
        """
        Construct and return a full scikit-learn pipeline for insurance charge prediction.

        Parameters
        ----------
        base_estimator:
            An uninitialised (or freshly initialised) scikit-learn regressor.
        handle_feature_skewness:
            When ``True``, chains ``PowerTransformer`` before ``StandardScaler``
            on numerical features.
        apply_log_target:
            When ``True``, wraps the inner pipeline in a
            ``TransformedTargetRegressor`` (``log1p`` / ``expm1``) to handle
            the right-skewed distribution of insurance ``charges``.
        include_interactions:
            When ``True``, prepends ``InsuranceFeatureEngineer`` to engineer
            interaction features before the ``ColumnTransformer``.

        Returns
        -------
        Pipeline | TransformedTargetRegressor
            A fully composed scikit-learn estimator ready for
            ``cross_validate()`` and ``.fit()``.
        """
        s = self.settings

        # ----------------------------------------------------------------
        # 1. Column definitions
        # ----------------------------------------------------------------
        base_numerical_cols = ["age", "bmi", "children"]
        interaction_cols = ["is_obese", "bmi_smoker_interaction", "age_bmi_interaction"]
        numerical_cols = (
            base_numerical_cols + interaction_cols
            if include_interactions
            else base_numerical_cols
        )

        binary_categorical_cols = ["sex", "smoker"]
        nominal_categorical_cols = ["region"]

        # ----------------------------------------------------------------
        # 2. Numerical sub-pipeline
        # ----------------------------------------------------------------
        if handle_feature_skewness:
            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "power_transform",
                        # pyrefly: ignore[redundant-cast]
                        PowerTransformer(method=cast(Literal["box-cox", "yeo-johnson"], s.POWER_TRANSFORM_METHOD)),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )
            scaling_desc = f"PowerTransformer({s.POWER_TRANSFORM_METHOD}) + StandardScaler"
        else:
            numerical_pipeline = Pipeline(
                steps=[("scaler", StandardScaler())]
            )
            scaling_desc = "StandardScaler"

        log.debug("Preprocessing numerical pipeline: %s", scaling_desc)

        # ----------------------------------------------------------------
        # 3. Categorical sub-pipelines
        # ----------------------------------------------------------------
        binary_encoder = OneHotEncoder(
            drop=s.OHE_DROP_STRATEGY,
            sparse_output=False,
            handle_unknown="ignore",
        )
        nominal_encoder = OneHotEncoder(
            drop=None,
            sparse_output=False,
            # pyrefly: ignore[redundant-cast]
            handle_unknown=cast(Literal["error", "ignore", "infrequent_if_exist"], s.OHE_HANDLE_UNKNOWN),
        )

        # ----------------------------------------------------------------
        # 4. ColumnTransformer
        # ----------------------------------------------------------------
        col_transformer = ColumnTransformer(
            transformers=[
                ("numerical", numerical_pipeline, numerical_cols),
                ("binary_cat", binary_encoder, binary_categorical_cols),
                ("nominal_cat", nominal_encoder, nominal_categorical_cols),
            ],
            remainder="drop",
        )

        # ----------------------------------------------------------------
        # 5. Full inner Pipeline
        # ----------------------------------------------------------------
        steps: list[tuple[str, Any]] = [
            (
                "feature_engineer",
                InsuranceFeatureEngineer(
                    bmi_obese_threshold=s.BMI_OBESE_THRESHOLD,
                    age_senior_threshold=s.AGE_SENIOR_RISK_THRESHOLD,
                    include_interactions=include_interactions,
                ),
            ),
            ("column_transformer", col_transformer),
            ("regressor", base_estimator),
        ]
        inner_pipeline = Pipeline(steps=steps)

        # ----------------------------------------------------------------
        # 6. Optional target transformation wrapper
        # ----------------------------------------------------------------
        if apply_log_target:
            final_estimator = TransformedTargetRegressor(
                regressor=inner_pipeline,
                func=np.log1p,
                inverse_func=np.expm1,
            )
            log.debug(
                "Pipeline wrapped in TransformedTargetRegressor(log1p/expm1) "
                "for target skew handling."
            )
        else:
            final_estimator = inner_pipeline

        log.info(
            "Built pipeline for %s: skewness_handling=%s, log_target=%s, interactions=%s.",
            type(base_estimator).__name__,
            handle_feature_skewness,
            apply_log_target,
            include_interactions,
        )
        return final_estimator
