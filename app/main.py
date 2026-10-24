"""
app/main.py
============
FastAPI application factory and entrypoint for the Insurance Cost Predictor
Microservice.

Usage
-----
    uvicorn app.main:app --reload

Public surface
--------------
- ``create_app() -> FastAPI``   — Application factory (testable).
- ``app``                       — Module-level singleton created by the factory.

Startup sequence
----------------
1. Log startup banner.
2. Check dataset availability.
3. Attempt to pre-load existing model artefacts.
4. If no artefacts exist but the dataset is available, run an initial
   training pass so the ``/predict`` and ``/model/metrics`` endpoints
   work out-of-the-box immediately after first launch.

No ``print()`` calls anywhere.  All logging via ``LoggerFactory``.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config.settings import get_settings
from app.controllers import eda_router, health_router, prediction_router, training_router
from app.exceptions.global_handler import register_exception_handlers
from app.services import get_dataset_service, get_prediction_service, get_training_service
from app.utils.logger import LoggerFactory, set_request_trace_id, get_request_trace_id


# ---------------------------------------------------------------------------
# Lifespan context manager — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Startup
    -------
    - Sets a SYSTEM-STARTUP trace ID for log correlation.
    - Logs the application banner.
    - Checks dataset availability.
    - Attempts to pre-load trained model artefacts from disk.
    - If no artefacts exist but the dataset is available, runs an initial
      training pass so ``/predict`` and ``/model/metrics`` work immediately.

    Shutdown
    --------
    - Logs a clean shutdown message.
    """
    # ------------------------------------------------------------------
    # STARTUP
    # ------------------------------------------------------------------
    set_request_trace_id("SYSTEM-STARTUP")
    log = LoggerFactory.get_logger(__name__)
    settings = get_settings()

    log.info(
        "=" * 72
    )
    log.info(
        "  %s  v%s  [%s]",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT.upper(),
    )
    log.info(
        "  API prefix : %s",
        settings.API_V1_PREFIX,
    )
    log.info(
        "  Swagger UI : http://127.0.0.1:8000/docs",
    )
    log.info(
        "=" * 72
    )

    dataset_service = get_dataset_service()
    prediction_service = get_prediction_service()
    training_service = get_training_service()

    # 1. Dataset availability
    dataset_available = dataset_service.check_dataset_availability()
    log.info("Dataset availability: %s — path='%s'.", dataset_available, settings.DATASET_FILE_PATH)

    # 2. Attempt to load existing artefacts
    artifacts_loaded = prediction_service.load_artifacts_into_memory()

    if artifacts_loaded:
        log.info(
            "Model artefacts pre-loaded — best_model='%s', last_trained_at='%s'.",
            prediction_service.active_best_model_name,
            prediction_service.last_trained_at,
        )
    elif dataset_available:
        # 3. Auto-train on first launch so the service is immediately usable
        log.info(
            "No existing artefacts found — running initial training pass "
            "so /predict and /model/metrics are immediately available."
        )
        try:
            training_service.train_and_evaluate()
            log.info(
                "Initial training pass complete — best_model='%s'.",
                prediction_service.active_best_model_name,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Initial auto-training failed (non-fatal): %s. "
                "Call POST /api/v1/model/train to train manually.",
                exc,
            )
    else:
        log.warning(
            "Dataset not found at '%s'. "
            "Upload the dataset and call POST /api/v1/model/train to train.",
            settings.DATASET_FILE_PATH,
        )

    log.info("%s startup complete.", settings.APP_NAME)

    yield

    # ------------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------------
    set_request_trace_id("SYSTEM-SHUTDOWN")
    log.info(
        "%s v%s shutting down gracefully.",
        settings.APP_NAME,
        settings.APP_VERSION,
    )


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application.

    Returns
    -------
    FastAPI
        Fully configured application instance ready for ``uvicorn``.
    """
    settings = get_settings()
    log = LoggerFactory.get_logger(__name__)

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Enterprise Insurance Cost Predictor Microservice — "
            "built with FastAPI, scikit-learn, and a native Resilience4j "
            "fault-tolerance pipeline (CircuitBreaker, RateLimiter, Bulkhead, "
            "TimeLimiter, Retry, and Actuarial Heuristic Fallback Engine)."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        debug=settings.DEBUG,
    )

    # ------------------------------------------------------------------
    # CORS Middleware
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-Id"],
    )

    # ------------------------------------------------------------------
    # SLF4J MDC Trace-ID Middleware
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def slf4j_mdc_trace_middleware(
        request: Request, call_next: object
    ) -> Response:
        """
        Per-request SLF4J MDC middleware.

        1. Reads ``X-Trace-Id`` from the incoming request header (or generates
           a new 8-char hex trace ID).
        2. Sets it in ``trace_id_context`` so every log line for this request
           carries the same trace ID.
        3. Logs the incoming request.
        4. Calls the next handler.
        5. Logs the outgoing response with latency.
        6. Attaches ``X-Trace-Id`` to the response headers.
        """
        _log = LoggerFactory.get_logger(__name__ + ".middleware")

        incoming_trace_id = request.headers.get("X-Trace-Id")
        trace_id = set_request_trace_id(incoming_trace_id)

        start_ns = time.perf_counter_ns()
        _log.info(
            "→ %s %s  [trace_id=%s]",
            request.method,
            request.url.path,
            trace_id,
        )

        response: Response = await call_next(request)  # type: ignore[operator]

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        _log.info(
            "← %s %s  status=%d  latency=%.3fms  [trace_id=%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            get_request_trace_id(),
        )

        response.headers["X-Trace-Id"] = trace_id
        return response

    # ------------------------------------------------------------------
    # Global exception handlers
    # ------------------------------------------------------------------
    register_exception_handlers(app)

    # ------------------------------------------------------------------
    # API routers — all mounted under /api/v1
    # ------------------------------------------------------------------
    prefix = settings.API_V1_PREFIX

    app.include_router(health_router, prefix=prefix)
    app.include_router(eda_router, prefix=prefix)
    app.include_router(training_router, prefix=prefix)
    app.include_router(prediction_router, prefix=prefix)

    log.info(
        "Routers mounted under prefix '%s': health, eda, training, prediction.",
        prefix,
    )

    # ------------------------------------------------------------------
    # Root welcome endpoint
    # ------------------------------------------------------------------
    @app.get(
        "/",
        tags=["Root"],
        summary="Welcome",
        description="Root redirect — points clients to Swagger UI and the health endpoint.",
        include_in_schema=True,
    )
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "service": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "status": "UP",
                "docs": "/docs",
                "health": f"{settings.API_V1_PREFIX}/health",
                "environment": settings.ENVIRONMENT,
            }
        )

    return app


# ---------------------------------------------------------------------------
# Module-level application singleton (used by uvicorn and TestClient)
# ---------------------------------------------------------------------------

app: FastAPI = create_app()
