"""
app/config/settings.py
======================
Central configuration module for the Insurance Cost Predictor Microservice.

All constants are loaded from the `.env` file via pydantic-settings `BaseSettings`.
No value is hardcoded in application source files — every threshold, path, ML
hyperparameter, resilience parameter, and log format is defined exclusively in `.env`.
"""

from __future__ import annotations

import sys
from typing import Literal
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Pydantic BaseSettings class that reads every configuration value from the
    `.env` file at the project root.  Type annotations enforce that the loaded
    values are coerced to the correct Python types at startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Group 1 — Application
    # ------------------------------------------------------------------
    APP_NAME: str = Field(..., description="Human-readable application name.")
    APP_VERSION: str = Field(..., description="Semantic version string.")
    API_V1_PREFIX: str = Field(..., description="URL prefix for all v1 endpoints.")
    ENVIRONMENT: str = Field(..., description="Runtime environment label (development / staging / production).")
    DEBUG: bool = Field(..., description="Enable debug mode (verbose FastAPI tracebacks).")
    DEFAULT_CURRENCY: str = Field(..., description="ISO 4217 currency code used in charge responses.")

    # ------------------------------------------------------------------
    # Group 2 — File Paths
    # ------------------------------------------------------------------
    DATASET_FILE_PATH: Path = Field(..., description="Relative path to the raw insurance CSV dataset.")
    ARTIFACTS_DIR: Path = Field(..., description="Directory where trained model artefacts are persisted.")
    BEST_PIPELINE_FILENAME: str = Field(..., description="Filename for the serialised best pipeline.")
    ALL_MODELS_FILENAME: str = Field(..., description="Filename for all evaluated model pipelines.")
    EVALUATION_REPORT_FILENAME: str = Field(..., description="Filename for the JSON evaluation report.")

    # ------------------------------------------------------------------
    # Group 3 — Domain & Preprocessing Thresholds
    # ------------------------------------------------------------------
    BMI_UNDERWEIGHT_THRESHOLD: float = Field(..., description="BMI below which a person is classified as underweight.")
    BMI_NORMAL_THRESHOLD: float = Field(..., description="BMI at or above which a person is classified as overweight.")
    BMI_OBESE_THRESHOLD: float = Field(..., description="BMI at or above which a person is classified as obese.")
    AGE_SENIOR_RISK_THRESHOLD: int = Field(..., description="Age at or above which a person is in the senior risk bracket.")
    SKEWNESS_HIGH_THRESHOLD: float = Field(..., description="Absolute skewness value above which power-transform is applied.")
    POWER_TRANSFORM_METHOD: Literal["box-cox", "yeo-johnson"] = Field(..., description="Scikit-learn PowerTransformer method (yeo-johnson or box-cox).")
    OHE_DROP_STRATEGY: str = Field(..., description="OneHotEncoder drop strategy (if_binary, first, None).")
    OHE_HANDLE_UNKNOWN: Literal["error", "ignore", "infrequent_if_exist"] = Field(..., description="OneHotEncoder unknown-category strategy (ignore or error).")

    # ------------------------------------------------------------------
    # Group 4 — DTO Validation Bounds
    # ------------------------------------------------------------------
    VALIDATION_AGE_MIN: int = Field(..., description="Minimum accepted age in prediction requests.")
    VALIDATION_AGE_MAX: int = Field(..., description="Maximum accepted age in prediction requests.")
    VALIDATION_BMI_MIN: float = Field(..., description="Minimum accepted BMI in prediction requests.")
    VALIDATION_BMI_MAX: float = Field(..., description="Maximum accepted BMI in prediction requests.")
    VALIDATION_CHILDREN_MIN: int = Field(..., description="Minimum accepted number of children.")
    VALIDATION_CHILDREN_MAX: int = Field(..., description="Maximum accepted number of children.")
    VALIDATION_CV_FOLDS_MIN: int = Field(..., description="Minimum cross-validation folds accepted in train requests.")
    VALIDATION_CV_FOLDS_MAX: int = Field(..., description="Maximum cross-validation folds accepted in train requests.")

    # ------------------------------------------------------------------
    # Group 5 — ML Defaults & Hyperparameters
    # ------------------------------------------------------------------
    DEFAULT_CV_FOLDS: int = Field(..., description="Default number of cross-validation folds.")
    DEFAULT_RANDOM_STATE: int = Field(..., description="Global random seed for reproducibility.")
    DEFAULT_DROP_DUPLICATES: bool = Field(..., description="Whether to drop duplicate rows during preprocessing.")
    DEFAULT_HANDLE_FEATURE_SKEW: bool = Field(..., description="Whether to apply power-transform to skewed numeric features.")
    DEFAULT_APPLY_LOG_TARGET: bool = Field(..., description="Whether to log-transform the target variable (charges).")
    DEFAULT_INCLUDE_INTERACTIONS: bool = Field(..., description="Whether to engineer interaction features (e.g., smoker×bmi).")

    # Linear Regression
    LR_FIT_INTERCEPT: bool = Field(..., description="Whether to fit an intercept term in Linear Regression.")

    # Random Forest
    RF_N_ESTIMATORS: int = Field(..., description="Number of trees in the Random Forest ensemble.")
    RF_MAX_DEPTH: int = Field(..., description="Maximum depth of each tree in the Random Forest.")
    RF_MIN_SAMPLES_SPLIT: int = Field(..., description="Minimum samples required to split an internal node (RF).")
    RF_MIN_SAMPLES_LEAF: int = Field(..., description="Minimum samples required at a leaf node (RF).")
    RF_N_JOBS: int = Field(..., description="Number of parallel jobs for Random Forest (-1 = all CPUs).")

    # Gradient Boosting
    GB_N_ESTIMATORS: int = Field(..., description="Number of boosting stages in Gradient Boosting.")
    GB_LEARNING_RATE: float = Field(..., description="Learning rate (shrinkage) for Gradient Boosting.")
    GB_MAX_DEPTH: int = Field(..., description="Maximum depth of individual regression estimators (GB).")
    GB_SUBSAMPLE: float = Field(..., description="Fraction of samples used for fitting each base learner (GB).")
    GB_MIN_SAMPLES_SPLIT: int = Field(..., description="Minimum samples required to split an internal node (GB).")

    # ------------------------------------------------------------------
    # Group 6 — Actuarial Fallback Engine Weights
    # ------------------------------------------------------------------
    FALLBACK_BASE_CHARGE: float = Field(..., description="Baseline insurance charge in the actuarial fallback formula.")
    FALLBACK_AGE_MULTIPLIER: float = Field(..., description="Per-year age contribution in the fallback formula.")
    FALLBACK_BMI_BASELINE: float = Field(..., description="Neutral BMI reference point in the fallback formula.")
    FALLBACK_EXCESS_BMI_MULTIPLIER: float = Field(..., description="Per-unit excess BMI charge in the fallback formula.")
    FALLBACK_CHILDREN_MULTIPLIER: float = Field(..., description="Per-child additional charge in the fallback formula.")
    FALLBACK_SMOKER_BASE_PENALTY: float = Field(..., description="Base penalty applied to smokers in the fallback formula.")
    FALLBACK_SMOKER_OBESE_SYNERGY_PENALTY: float = Field(..., description="Extra penalty for smokers who are also obese.")

    # ------------------------------------------------------------------
    # Group 7 — Resilience4j Circuit Breaker
    # ------------------------------------------------------------------
    # predict
    CB_PREDICT_FAILURE_RATE_THRESHOLD: float = Field(..., description="Failure-rate % that trips the predict circuit breaker.")
    CB_PREDICT_SLIDING_WINDOW_SIZE: int = Field(..., description="Sliding window size (calls) for predict circuit breaker.")
    CB_PREDICT_MIN_NUMBER_OF_CALLS: int = Field(..., description="Minimum calls before failure rate is evaluated (predict).")
    CB_PREDICT_WAIT_DURATION_IN_OPEN_STATE_SEC: float = Field(..., description="Seconds the predict CB stays OPEN before transitioning to HALF_OPEN.")
    CB_PREDICT_PERMITTED_CALLS_IN_HALF_OPEN: int = Field(..., description="Probe calls allowed in HALF_OPEN state (predict).")

    # train
    CB_TRAIN_FAILURE_RATE_THRESHOLD: float = Field(..., description="Failure-rate % that trips the train circuit breaker.")
    CB_TRAIN_SLIDING_WINDOW_SIZE: int = Field(..., description="Sliding window size (calls) for train circuit breaker.")
    CB_TRAIN_MIN_NUMBER_OF_CALLS: int = Field(..., description="Minimum calls before failure rate is evaluated (train).")
    CB_TRAIN_WAIT_DURATION_IN_OPEN_STATE_SEC: float = Field(..., description="Seconds the train CB stays OPEN before transitioning to HALF_OPEN.")
    CB_TRAIN_PERMITTED_CALLS_IN_HALF_OPEN: int = Field(..., description="Probe calls allowed in HALF_OPEN state (train).")

    # eda
    CB_EDA_FAILURE_RATE_THRESHOLD: float = Field(..., description="Failure-rate % that trips the eda circuit breaker.")
    CB_EDA_SLIDING_WINDOW_SIZE: int = Field(..., description="Sliding window size (calls) for eda circuit breaker.")
    CB_EDA_MIN_NUMBER_OF_CALLS: int = Field(..., description="Minimum calls before failure rate is evaluated (eda).")
    CB_EDA_WAIT_DURATION_IN_OPEN_STATE_SEC: float = Field(..., description="Seconds the eda CB stays OPEN before transitioning to HALF_OPEN.")
    CB_EDA_PERMITTED_CALLS_IN_HALF_OPEN: int = Field(..., description="Probe calls allowed in HALF_OPEN state (eda).")

    # ------------------------------------------------------------------
    # Group 8 — Resilience4j Rate Limiter
    # ------------------------------------------------------------------
    RL_PREDICT_LIMIT_FOR_PERIOD: int = Field(..., description="Max permitted calls per refresh period (predict).")
    RL_PREDICT_REFRESH_PERIOD_SEC: float = Field(..., description="Refresh period in seconds for predict rate limiter.")

    RL_TRAIN_LIMIT_FOR_PERIOD: int = Field(..., description="Max permitted calls per refresh period (train).")
    RL_TRAIN_REFRESH_PERIOD_SEC: float = Field(..., description="Refresh period in seconds for train rate limiter.")

    RL_EDA_LIMIT_FOR_PERIOD: int = Field(..., description="Max permitted calls per refresh period (eda).")
    RL_EDA_REFRESH_PERIOD_SEC: float = Field(..., description="Refresh period in seconds for eda rate limiter.")

    # ------------------------------------------------------------------
    # Group 9 — Resilience4j Bulkhead
    # ------------------------------------------------------------------
    BH_PREDICT_MAX_CONCURRENT_CALLS: int = Field(..., description="Maximum concurrent calls allowed (predict bulkhead).")
    BH_PREDICT_MAX_WAIT_DURATION_SEC: float = Field(..., description="Max seconds to wait for a bulkhead permit (predict).")

    BH_TRAIN_MAX_CONCURRENT_CALLS: int = Field(..., description="Maximum concurrent calls allowed (train bulkhead).")
    BH_TRAIN_MAX_WAIT_DURATION_SEC: float = Field(..., description="Max seconds to wait for a bulkhead permit (train).")

    BH_EDA_MAX_CONCURRENT_CALLS: int = Field(..., description="Maximum concurrent calls allowed (eda bulkhead).")
    BH_EDA_MAX_WAIT_DURATION_SEC: float = Field(..., description="Max seconds to wait for a bulkhead permit (eda).")

    # ------------------------------------------------------------------
    # Group 10 — Resilience4j TimeLimiter
    # ------------------------------------------------------------------
    TL_PREDICT_TIMEOUT_SEC: float = Field(..., description="Max execution seconds before TimeLimiter cancels predict.")
    TL_TRAIN_TIMEOUT_SEC: float = Field(..., description="Max execution seconds before TimeLimiter cancels train.")
    TL_EDA_TIMEOUT_SEC: float = Field(..., description="Max execution seconds before TimeLimiter cancels eda.")

    # ------------------------------------------------------------------
    # Group 11 — Resilience4j Retry & Fallback Toggles
    # ------------------------------------------------------------------
    RETRY_MAX_ATTEMPTS: int = Field(..., description="Total attempts (1 initial + N-1 retries) before giving up.")
    RETRY_WAIT_DURATION_SEC: float = Field(..., description="Base wait between retry attempts in seconds.")
    RETRY_BACKOFF_MULTIPLIER: float = Field(..., description="Exponential backoff multiplier applied to the wait duration.")
    ENABLE_FALLBACK_ON_PREDICT: bool = Field(..., description="Toggle the actuarial fallback for the predict endpoint.")
    ENABLE_FALLBACK_ON_METRICS: bool = Field(..., description="Toggle the fallback path for the metrics/health endpoint.")

    # ------------------------------------------------------------------
    # Group 12 — SLF4J Console Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = Field(..., description="Root logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    LOG_FORMAT: str = Field(..., description="Python logging format string with MDC trace_id injection.")
    LOG_DATE_FORMAT: str = Field(..., description="strftime-compatible date format for log records.")

    # ------------------------------------------------------------------
    # Computed path properties
    # ------------------------------------------------------------------

    @property
    def best_pipeline_path(self) -> Path:
        """Absolute path to the serialised best pipeline artefact."""
        return self.ARTIFACTS_DIR / self.BEST_PIPELINE_FILENAME

    @property
    def all_models_path(self) -> Path:
        """Absolute path to the serialised all-models artefact."""
        return self.ARTIFACTS_DIR / self.ALL_MODELS_FILENAME

    @property
    def evaluation_report_path(self) -> Path:
        """Absolute path to the JSON evaluation report artefact."""
        return self.ARTIFACTS_DIR / self.EVALUATION_REPORT_FILENAME


@lru_cache()
def get_settings() -> Settings:
    """
    Return the singleton Settings instance (cached after first call).

    Side-effect: ensures ARTIFACTS_DIR exists on first access so that
    downstream serialisation code never has to create the directory itself.
    """
    settings = Settings()
    settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return settings
