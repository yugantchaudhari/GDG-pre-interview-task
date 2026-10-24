"""
app/controllers/health_controller.py
======================================
Health & Resilience4j Telemetry controller.

Exposes a single ``GET /health`` endpoint that reports:
- Dataset file availability on disk.
- Model training state (artefact presence + in-memory load status).
- A full Resilience4j registry snapshot (CB, RL, BH, TL, Retry for every profile).

No ``print()`` calls anywhere.  Uses ``LoggerFactory`` for all logging.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dto.health_dto import HealthResponseDTO
from app.resilience.registry import resilience_registry
from app.services import DatasetService, PredictionService, get_dataset_service, get_prediction_service
from app.utils.logger import LoggerFactory

log = LoggerFactory.get_logger(__name__)

router = APIRouter(tags=["Health & Resilience4j Telemetry"])


@router.get(
    "/health",
    response_model=HealthResponseDTO,
    summary="Service health check",
    description=(
        "Returns overall service status, dataset availability, model training state, "
        "and a live snapshot of all Resilience4j component metrics."
    ),
)
async def get_health(
    dataset_service: DatasetService = Depends(get_dataset_service),
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> HealthResponseDTO:
    """
    Perform a comprehensive health check of all microservice subsystems.

    Returns
    -------
    HealthResponseDTO
        Live health snapshot including Resilience4j registry state.
    """
    log.info("GET /health — performing subsystem health check.")

    # 1. Check dataset availability
    dataset_available = dataset_service.check_dataset_availability()
    dataset_path = str(dataset_service.settings.DATASET_FILE_PATH)

    # 2. Check model artifact state — attempt lazy load if not yet in memory
    model_trained = prediction_service.model_trained
    if model_trained and prediction_service.active_best_model_name is None:
        prediction_service.load_artifacts_into_memory()

    log.info(
        "Health check — dataset_available=%s, model_trained=%s, "
        "active_model='%s', last_trained_at='%s'.",
        dataset_available,
        model_trained,
        prediction_service.active_best_model_name,
        prediction_service.last_trained_at,
    )

    # 3. Pull full Resilience4j registry snapshot
    registry_snapshot = resilience_registry.get_full_registry_snapshot()

    overall_status = "UP" if dataset_available else "DEGRADED"

    return HealthResponseDTO(
        status=overall_status,
        dataset_available=dataset_available,
        dataset_path=dataset_path,
        model_trained=model_trained,
        active_best_model=prediction_service.active_best_model_name,
        last_trained_at=prediction_service.last_trained_at,
        resilience4j_registry=registry_snapshot,
    )
