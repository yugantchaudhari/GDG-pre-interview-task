"""
app/controllers/training_controller.py
========================================
Model Training & Cross-Validation controller.

Exposes:
- ``POST /model/train``   — Triggers a full multi-model cross-validated
                            training run (LR, RF, GB) and returns the
                            ranked evaluation leaderboard.
- ``GET  /model/metrics`` — Returns the last persisted evaluation report
                            with actuarial fallback when no model is trained.

Both endpoints are protected by the ``@resilience4j_pipeline`` decorator.
The metrics endpoint uses the actuarial fallback when
``ENABLE_FALLBACK_ON_METRICS=True`` in ``.env``.

No ``print()`` calls anywhere.  Uses ``LoggerFactory`` for all logging.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends

from app.config.settings import get_settings
from app.dto.training_dto import TrainingRequestDTO, TrainingResponseDTO
from app.resilience.decorators import resilience4j_pipeline
from app.services import (
    FallbackService,
    TrainingService,
    get_fallback_service,
    get_training_service,
)
from app.utils.logger import LoggerFactory

log = LoggerFactory.get_logger(__name__)

router = APIRouter(prefix="/model", tags=["2. Model Training & Cross-Validation"])


@router.post(
    "/train",
    response_model=TrainingResponseDTO,
    summary="Train & evaluate models",
    description=(
        "Runs a full cross-validated multi-model training run: "
        "LinearRegression, RandomForestRegressor, and GradientBoostingRegressor. "
        "Models are ranked by mean R² across K folds (lowest RMSE on tie). "
        "The best pipeline is persisted to disk and hot-reloaded into memory. "
        "Protected by the Resilience4j 'train' profile."
    ),
)
async def train_model(
    request: TrainingRequestDTO = Body(default_factory=TrainingRequestDTO),
    training_service: TrainingService = Depends(get_training_service),
) -> TrainingResponseDTO:
    """
    Train all candidate models via K-fold cross-validation and return the leaderboard.

    Parameters
    ----------
    request:
        Training configuration (cv_folds, preprocessing flags, etc.).
        Defaults to ``.env``-configured values when body is empty.

    Returns
    -------
    TrainingResponseDTO
        Ranked leaderboard, preprocessing summary, and artifact paths.
    """
    log.info(
        "POST /model/train — cv_folds=%d, drop_duplicates=%s, "
        "log_target=%s, interactions=%s.",
        request.cv_folds,
        request.drop_duplicates,
        request.apply_log_transform_on_target,
        request.include_bmi_smoker_interaction,
    )

    @resilience4j_pipeline(profile="train")
    async def _protected_train() -> TrainingResponseDTO:
        return training_service.train_and_evaluate(request)

    result: TrainingResponseDTO = await _protected_train()
    return result


@router.get(
    "/metrics",
    response_model=TrainingResponseDTO,
    summary="Retrieve saved evaluation metrics",
    description=(
        "Returns the persisted evaluation report from the last successful training run. "
        "When no model has been trained yet OR when a Resilience4j fault occurs AND "
        "``ENABLE_FALLBACK_ON_METRICS=True``, the actuarial fallback engine returns a "
        "degraded but valid response. "
        "Protected by the Resilience4j 'eda' profile."
    ),
)
async def get_metrics(
    training_service: TrainingService = Depends(get_training_service),
    fallback_service: FallbackService = Depends(get_fallback_service),
) -> TrainingResponseDTO:
    """
    Load and return the persisted model evaluation report.

    Returns
    -------
    TrainingResponseDTO
        Last saved leaderboard and metrics, or a graceful degraded fallback.
    """
    settings = get_settings()
    log.info(
        "GET /model/metrics — fallback_enabled=%s.",
        settings.ENABLE_FALLBACK_ON_METRICS,
    )

    def _metrics_fallback(reason: str, resilience_metadata: dict, **kwargs) -> TrainingResponseDTO:  # type: ignore[type-arg]
        return fallback_service.get_metrics_fallback(
            reason=reason,
            resilience_metadata=resilience_metadata,
        )

    @resilience4j_pipeline(profile="eda", fallback_callable=_metrics_fallback)
    async def _protected_metrics() -> TrainingResponseDTO:
        return training_service.get_saved_metrics()

    result: TrainingResponseDTO = await _protected_metrics()
    return result
