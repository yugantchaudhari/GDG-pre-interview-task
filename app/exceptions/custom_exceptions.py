"""
app/exceptions/custom_exceptions.py
====================================
Custom exception hierarchy for the Insurance Cost Predictor Microservice.

Design principles
-----------------
* Every exception carries a machine-readable ``error_code``, an HTTP
  ``status_code``, and an optional ``pattern`` that identifies which
  Resilience4j pattern raised it — mirroring the structured-error model of
  Spring Boot's ``ProblemDetail`` / RFC 7807.
* ``BaseApiException`` is the single root for all application exceptions,
  making it trivial to write a FastAPI exception handler that intercepts
  every application-level error.
* Resilience4j exceptions carry the originating pattern name so that the
  global handler can include it in the error response body.
* No ``print()`` calls anywhere in this module.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Root exception
# ---------------------------------------------------------------------------

class BaseApiException(Exception):
    """
    Root exception for all application-layer errors.

    Attributes
    ----------
    message:
        Human-readable description of the error.
    error_code:
        Machine-readable error code (SCREAMING_SNAKE_CASE) for client
        programmatic handling.
    status_code:
        HTTP status code that the global exception handler will use when
        serialising this exception to a response.
    pattern:
        Optional name of the Resilience4j pattern that raised this error
        (e.g. ``"Resilience4j.CircuitBreaker"``).  ``None`` for domain errors.
    instance_name:
        Optional name of the specific resilience instance (e.g. the circuit
        breaker name) that raised the error.
    details:
        Optional free-form dictionary of additional context included in the
        error response body.
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int,
        pattern: str | None = None,
        instance_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.pattern = pattern
        self.instance_name = instance_name
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this exception."""
        payload: dict[str, Any] = {
            "error_code": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
        }
        if self.pattern:
            payload["pattern"] = self.pattern
        if self.instance_name:
            payload["instance_name"] = self.instance_name
        if self.details:
            payload["details"] = self.details
        return payload

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"error_code={self.error_code!r}, "
            f"status_code={self.status_code}, "
            f"message={self.message!r})"
        )


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------

class DatasetNotFoundException(BaseApiException):
    """
    Raised when the configured dataset file does not exist on disk.

    HTTP 404 — the client should ensure the dataset has been uploaded or
    the ``DATASET_FILE_PATH`` setting is correct.
    """

    def __init__(
        self,
        message: str = "The insurance dataset file could not be found.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="DATASET_NOT_FOUND",
            status_code=404,
            details=details,
        )


class InvalidDatasetSchemaException(BaseApiException):
    """
    Raised when the dataset is found but does not contain the expected columns
    or contains data that cannot be coerced to the required types.

    HTTP 400 — the client/operator must supply a valid dataset.
    """

    def __init__(
        self,
        message: str = "The dataset schema is invalid or missing required columns.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="INVALID_DATASET_SCHEMA",
            status_code=400,
            details=details,
        )


class ModelNotTrainedException(BaseApiException):
    """
    Raised when a prediction is requested but no trained pipeline artefact
    exists in ``ARTIFACTS_DIR``.

    HTTP 400 — the client must trigger a training run via ``/train`` first.
    """

    def __init__(
        self,
        message: str = "No trained model pipeline found. Please call the /train endpoint first.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="MODEL_NOT_TRAINED",
            status_code=400,
            details=details,
        )


class InvalidModelOverrideException(BaseApiException):
    """
    Raised when the client requests prediction using a model name that does
    not exist among the trained models stored in ``ALL_MODELS_FILENAME``.

    HTTP 400 — the client should check available model names from the
    evaluation report.
    """

    def __init__(
        self,
        message: str = "The requested model override name is not a valid trained model.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="INVALID_MODEL_OVERRIDE",
            status_code=400,
            details=details,
        )


# ---------------------------------------------------------------------------
# Resilience4j exceptions
# ---------------------------------------------------------------------------

class CircuitBreakerOpenException(BaseApiException):
    """
    Raised by the ``CircuitBreaker`` when it is in the ``OPEN`` state and
    refusing requests to protect the downstream service.

    HTTP 503 — the client should retry after the wait duration has elapsed.
    """

    def __init__(
        self,
        instance_name: str = "unknown",
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        resolved_message = message or (
            f"Circuit breaker '{instance_name}' is OPEN. "
            "Requests are temporarily blocked to protect the service."
        )
        super().__init__(
            message=resolved_message,
            error_code="CIRCUIT_BREAKER_OPEN",
            status_code=503,
            pattern="Resilience4j.CircuitBreaker",
            instance_name=instance_name,
            details=details,
        )


class RateLimitExceededException(BaseApiException):
    """
    Raised by the ``RateLimiter`` when the caller has exceeded the configured
    ``limit_for_period`` within the current ``refresh_period_sec``.

    HTTP 429 — the client should back off and retry after ``retry_after_seconds``.
    """

    def __init__(
        self,
        instance_name: str = "unknown",
        retry_after_seconds: float = 0.0,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        resolved_message = message or (
            f"Rate limit exceeded for '{instance_name}'. "
            f"Retry after {retry_after_seconds:.2f} seconds."
        )
        extra_details: dict[str, Any] = {"retry_after_seconds": retry_after_seconds}
        if details:
            extra_details.update(details)
        super().__init__(
            message=resolved_message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            pattern="Resilience4j.RateLimiter",
            instance_name=instance_name,
            details=extra_details,
        )
        self.retry_after_seconds = retry_after_seconds


class BulkheadFullException(BaseApiException):
    """
    Raised by the ``Bulkhead`` when all concurrency permits are exhausted and
    the caller waited longer than ``max_wait_duration_sec``.

    HTTP 429 — the service is currently at maximum concurrency; retry later.
    """

    def __init__(
        self,
        instance_name: str = "unknown",
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        resolved_message = message or (
            f"Bulkhead '{instance_name}' is full. "
            "Maximum concurrency reached; please retry later."
        )
        super().__init__(
            message=resolved_message,
            error_code="BULKHEAD_FULL",
            status_code=429,
            pattern="Resilience4j.Bulkhead",
            instance_name=instance_name,
            details=details,
        )


class TimeLimiterTimeoutException(BaseApiException):
    """
    Raised by the ``TimeLimiter`` when the protected coroutine does not
    complete within ``timeout_sec``.

    HTTP 504 — the upstream computation exceeded its time budget.
    """

    def __init__(
        self,
        instance_name: str = "unknown",
        timeout_sec: float = 0.0,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        resolved_message = message or (
            f"TimeLimiter '{instance_name}' timed out after {timeout_sec:.2f}s. "
            "The operation did not complete within the permitted window."
        )
        extra_details: dict[str, Any] = {"timeout_sec": timeout_sec}
        if details:
            extra_details.update(details)
        super().__init__(
            message=resolved_message,
            error_code="TIME_LIMITER_TIMEOUT",
            status_code=504,
            pattern="Resilience4j.TimeLimiter",
            instance_name=instance_name,
            details=extra_details,
        )
        self.timeout_sec = timeout_sec
