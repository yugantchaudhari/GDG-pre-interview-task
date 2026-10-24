"""
app/dto/resilience_dto.py
==========================
Pydantic v2 Data Transfer Objects for Resilience4j telemetry and error
responses.

These models mirror the shape of ``resilience_registry.get_full_registry_snapshot()``
so that the ``/api/v1/health`` endpoint can validate and serialise the live
circuit-breaker / rate-limiter / bulkhead state with zero raw-dict leakage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Resilience metadata injected into every successful response
# ---------------------------------------------------------------------------

class ResilienceMetadataDTO(BaseModel):
    """Per-request Resilience4j execution telemetry."""

    circuit_breaker_name: str = Field(
        ..., description="Name of the circuit breaker instance that guarded this request."
    )
    circuit_breaker_state: str = Field(
        ..., description="State of the circuit breaker at response time (CLOSED/OPEN/HALF_OPEN)."
    )
    fallback_triggered: bool = Field(
        default=False,
        description="Whether the actuarial fallback engine was invoked for this response.",
    )
    fallback_reason: str | None = Field(
        default=None,
        description="Human-readable reason the fallback was triggered (None when not triggered).",
    )
    retry_attempts_used: int = Field(
        default=0,
        description="Number of retry attempts consumed (0 = first attempt succeeded).",
    )
    bulkhead_available_permits: int = Field(
        ..., description="Remaining concurrent-call permits in the bulkhead at response time."
    )
    rate_limit_remaining: int = Field(
        ..., description="Remaining rate-limit permits for the current refresh window."
    )
    execution_time_ms: float = Field(
        ..., description="Total wall-clock time consumed by the pipeline execution in milliseconds."
    )


# ---------------------------------------------------------------------------
# Registry snapshot component DTOs (used by /health)
# ---------------------------------------------------------------------------

class CircuitBreakerSnapshotDTO(BaseModel):
    """Point-in-time metrics snapshot of a single CircuitBreaker instance."""

    name: str
    state: str
    failure_rate_percent: float
    buffered_calls: int
    failed_calls: int
    wait_duration_open_state_sec: float
    permitted_calls_in_half_open: int


class RateLimiterSnapshotDTO(BaseModel):
    """Point-in-time metrics snapshot of a single RateLimiter instance."""

    name: str
    limit_for_period: int
    limit_refresh_period_seconds: float
    available_permissions: int


class BulkheadSnapshotDTO(BaseModel):
    """Point-in-time metrics snapshot of a single Bulkhead instance."""

    name: str
    max_concurrent_calls: int
    available_concurrent_calls: int
    max_wait_duration_sec: float


class TimeLimiterSnapshotDTO(BaseModel):
    """Configuration snapshot of a single TimeLimiter instance."""

    name: str
    timeout_sec: float


class RetrySnapshotDTO(BaseModel):
    """Configuration snapshot of a single Retry instance."""

    name: str
    max_attempts: int
    wait_duration_sec: float
    backoff_multiplier: float


class ProfileSnapshotDTO(BaseModel):
    """All Resilience4j component snapshots for a single endpoint profile."""

    circuit_breaker: CircuitBreakerSnapshotDTO
    rate_limiter: RateLimiterSnapshotDTO
    bulkhead: BulkheadSnapshotDTO
    time_limiter: TimeLimiterSnapshotDTO
    retry: RetrySnapshotDTO


class ResilienceRegistrySnapshotDTO(BaseModel):
    """
    Top-level snapshot of the entire Resilience4j registry.

    Keys are profile names: ``"predict"``, ``"train"``, ``"eda"``.
    """

    predict: ProfileSnapshotDTO
    train: ProfileSnapshotDTO
    eda: ProfileSnapshotDTO


# ---------------------------------------------------------------------------
# Structured API error response
# ---------------------------------------------------------------------------

class ErrorResponseDTO(BaseModel):
    """
    RFC 7807-inspired structured error body returned by the global exception
    handler for all ``BaseApiException`` subclasses.
    """

    error_code: str = Field(
        ..., description="Machine-readable SCREAMING_SNAKE_CASE error identifier."
    )
    pattern: str | None = Field(
        default=None,
        description="Resilience4j pattern that raised the error (e.g. Resilience4j.CircuitBreaker).",
    )
    instance_name: str | None = Field(
        default=None,
        description="Name of the specific resilience instance that raised the error.",
    )
    state: str | None = Field(
        default=None,
        description="Circuit-breaker state at the time of the error (for CB errors only).",
    )
    message: str = Field(..., description="Human-readable error description.")
    retry_after_seconds: float | None = Field(
        default=None,
        description="Seconds the client should wait before retrying (for rate-limit errors).",
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional free-form additional context from the raising exception.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp when this error was generated.",
    )
