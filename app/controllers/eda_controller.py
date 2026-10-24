"""
app/controllers/eda_controller.py
====================================
Exploratory Data Analysis (EDA) controller.

Exposes ``GET /eda/summary`` which generates a comprehensive statistical
profile of the insurance dataset.  The endpoint is protected by the
``@resilience4j_pipeline(profile="eda")`` decorator.

Note on decorator compatibility:
---------------------------------
To preserve FastAPI's OpenAPI schema generation and dependency injection,
the resilience pipeline is applied around an inner async helper function
rather than the FastAPI route handler itself.  The route handler delegates
to the helper and returns its result directly.

No ``print()`` calls anywhere.  Uses ``LoggerFactory`` for all logging.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dto.eda_dto import EdaSummaryResponseDTO
from app.resilience.decorators import resilience4j_pipeline
from app.services import EdaService, get_eda_service
from app.utils.logger import LoggerFactory

log = LoggerFactory.get_logger(__name__)

router = APIRouter(prefix="/eda", tags=["1. Exploratory Data Analysis (EDA)"])


@router.get(
    "/summary",
    response_model=EdaSummaryResponseDTO,
    summary="Generate EDA summary",
    description=(
        "Computes a comprehensive statistical summary of the insurance dataset: "
        "row/column counts, duplicate detection, per-feature skewness flags, "
        "categorical value distributions, and actuarial domain insights "
        "(smoker cost multiplier, Pearson correlations). "
        "Protected by Resilience4j CircuitBreaker / RateLimiter / Bulkhead / TimeLimiter / Retry "
        "on the 'eda' profile."
    ),
)
async def get_eda_summary(
    eda_service: EdaService = Depends(get_eda_service),
) -> EdaSummaryResponseDTO:
    """
    Generate and return the EDA summary, protected by the Resilience4j pipeline.

    Returns
    -------
    EdaSummaryResponseDTO
        Fully populated EDA response with resilience telemetry injected.
    """
    log.info("GET /eda/summary — generating EDA summary (resilience4j profile: eda).")

    @resilience4j_pipeline(profile="eda")
    async def _protected_eda() -> EdaSummaryResponseDTO:
        return eda_service.generate_eda_summary()

    result: EdaSummaryResponseDTO = await _protected_eda()
    return result
