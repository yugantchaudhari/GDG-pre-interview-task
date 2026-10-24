"""
app/controllers/prediction_controller.py
==========================================
Cost Prediction & Risk Profiling controller.

Exposes ``POST /predict`` which performs real-time inference using the
best trained pipeline (or a named model override).

The endpoint is protected by ``@resilience4j_pipeline(profile="predict")``
with an actuarial fallback callable that invokes
``FallbackService.predict_charge_fallback`` when:
- The ML pipeline raises any exception, AND
- ``ENABLE_FALLBACK_ON_PREDICT=True`` in ``.env``.

The fallback guarantees a valid ``PredictionResponseDTO`` is always returned,
with ``model_used == "ActuarialHeuristicFallbackEngine"`` and
``resilience_metadata.fallback_triggered == True``.

No ``print()`` calls anywhere.  Uses ``LoggerFactory`` for all logging.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dto.prediction_dto import PredictionRequestDTO, PredictionResponseDTO
from app.resilience.decorators import resilience4j_pipeline
from app.services import (
    FallbackService,
    PredictionService,
    get_fallback_service,
    get_prediction_service,
)
from app.utils.logger import LoggerFactory

log = LoggerFactory.get_logger(__name__)

router = APIRouter(tags=["3. Cost Prediction & Risk Profiling"])


@router.post(
    "/predict",
    response_model=PredictionResponseDTO,
    summary="Predict insurance cost",
    description=(
        "Predicts the annual insurance charge for a policyholder using the best "
        "trained ML pipeline. Returns the predicted charge, BMI classification, "
        "actuarial risk tier (LOW_RISK / MODERATE_RISK / HIGH_RISK), key cost drivers, "
        "and Resilience4j execution telemetry. "
        "When the primary ML path is unavailable and ``ENABLE_FALLBACK_ON_PREDICT=True``, "
        "the Actuarial Heuristic Fallback Engine computes a degraded-but-valid estimate. "
        "Protected by the Resilience4j 'predict' profile."
    ),
)
async def predict_charge(
    request: PredictionRequestDTO,
    prediction_service: PredictionService = Depends(get_prediction_service),
    fallback_service: FallbackService = Depends(get_fallback_service),
) -> PredictionResponseDTO:
    """
    Predict the annual insurance cost for a single policyholder.

    Parameters
    ----------
    request:
        Validated policyholder features (age, sex, bmi, children, smoker, region).

    Returns
    -------
    PredictionResponseDTO
        Predicted charge, risk assessment, and Resilience4j telemetry.
    """
    log.info(
        "POST /predict — age=%d bmi=%.1f smoker=%s region=%s (model_override=%s).",
        request.age,
        request.bmi,
        request.smoker,
        request.region,
        request.model_override,
    )

    def _predict_fallback(
        reason: str,
        resilience_metadata: dict,  # type: ignore[type-arg]
        **kwargs,  # type: ignore[type-arg]
    ) -> PredictionResponseDTO:
        return fallback_service.predict_charge_fallback(
            request=request,
            reason=reason,
            resilience_metadata=resilience_metadata,
        )

    @resilience4j_pipeline(profile="predict", fallback_callable=_predict_fallback)
    async def _protected_predict() -> PredictionResponseDTO:
        return prediction_service.predict_charge(request)

    result: PredictionResponseDTO = await _protected_predict()
    return result
