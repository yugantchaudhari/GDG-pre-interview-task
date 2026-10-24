"""
app/exceptions/global_handler.py
==================================
Global FastAPI exception handlers for the Insurance Cost Predictor Microservice.

Registered via ``register_exception_handlers(app)`` called in ``app/main.py``.

Handles:
- ``BaseApiException`` subclasses (all application-layer errors including
  Resilience4j control-flow exceptions).
- ``RequestValidationError`` (Pydantic / FastAPI 422 validation failures).
- Generic ``Exception`` (catch-all 500 Internal Server Error).

No ``print()`` calls anywhere in this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.dto.resilience_dto import ErrorResponseDTO
from app.exceptions.custom_exceptions import BaseApiException
from app.utils.logger import LoggerFactory

log = LoggerFactory.get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """
    Attach all global exception handlers to the FastAPI *app* instance.

    Parameters
    ----------
    app:
        The FastAPI application returned by ``create_app()``.
    """

    # -----------------------------------------------------------------------
    # Handler 1: BaseApiException — all domain & Resilience4j errors
    # -----------------------------------------------------------------------
    @app.exception_handler(BaseApiException)
    async def base_api_exception_handler(
        request: Request, exc: BaseApiException
    ) -> JSONResponse:
        retry_after_seconds = getattr(exc, "retry_after_seconds", None)

        # Choose log level based on HTTP status code severity
        if exc.status_code >= 500:
            log.error(
                "BaseApiException [%s] error_code='%s' pattern='%s' status=%d — %s",
                exc.__class__.__name__,
                exc.error_code,
                exc.pattern,
                exc.status_code,
                exc.message,
            )
        else:
            log.warning(
                "BaseApiException [%s] error_code='%s' pattern='%s' status=%d — %s",
                exc.__class__.__name__,
                exc.error_code,
                exc.pattern,
                exc.status_code,
                exc.message,
            )

        error_dto = ErrorResponseDTO(
            error_code=exc.error_code,
            pattern=exc.pattern,
            instance_name=exc.instance_name,
            state="OPEN" if exc.error_code == "CIRCUIT_BREAKER_OPEN" else None,
            message=exc.message,
            retry_after_seconds=retry_after_seconds,
            details=exc.details if exc.details else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_dto.model_dump(mode="json"),
        )

    # -----------------------------------------------------------------------
    # Handler 2: RequestValidationError — Pydantic/FastAPI 422
    # -----------------------------------------------------------------------
    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        log.warning(
            "RequestValidationError on %s %s — %d validation error(s).",
            request.method,
            request.url.path,
            len(exc.errors()),
        )

        error_dto = ErrorResponseDTO(
            error_code="VALIDATION_ERROR",
            pattern=None,
            instance_name=None,
            state=None,
            message="Request payload validation failed.",
            retry_after_seconds=None,
            details={"errors": exc.errors()},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return JSONResponse(
            status_code=422,
            content=error_dto.model_dump(mode="json"),
        )

    # -----------------------------------------------------------------------
    # Handler 3: Generic Exception — catch-all 500
    # -----------------------------------------------------------------------
    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        log.error(
            "Unhandled exception on %s %s — %s: %s",
            request.method,
            request.url.path,
            type(exc).__name__,
            exc,
            exc_info=True,
        )

        error_dto = ErrorResponseDTO(
            error_code="INTERNAL_SERVER_ERROR",
            pattern=None,
            instance_name=None,
            state=None,
            message="An unexpected internal server error occurred.",
            retry_after_seconds=None,
            details={"error_type": type(exc).__name__},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return JSONResponse(
            status_code=500,
            content=error_dto.model_dump(mode="json"),
        )

    log.info("Global exception handlers registered successfully.")
