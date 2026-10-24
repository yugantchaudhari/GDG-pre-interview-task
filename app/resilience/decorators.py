"""
app/resilience/decorators.py
==============================
``@resilience4j_pipeline`` — the master decorator that wraps FastAPI endpoint
handlers (or service functions) with the full Resilience4j execution chain.

Execution order (outer → inner, matching Spring Boot's default)
--------------------------------------------------------------
    Fallback → CircuitBreaker → RateLimiter → TimeLimiter → Bulkhead → Retry → Target

The decorator supports both ``async def`` and synchronous ``def`` targets.
Sync targets are executed inside the ``TimeLimiter`` via ``asyncio.to_thread``
so that the event loop is not blocked.

Resilience metadata
-------------------
After each call (successful or fallback) the decorator builds a
``resilience_metadata`` dictionary containing:
- ``circuit_breaker_name``
- ``circuit_breaker_state``
- ``fallback_triggered``        (bool)
- ``fallback_reason``           (str | None)
- ``retry_attempts_used``       (int)
- ``bulkhead_available_permits``(int)
- ``rate_limit_remaining``      (int)
- ``execution_time_ms``         (float)

This metadata is automatically injected into the response when:
- The response is a plain ``dict``
- The response is a Pydantic ``BaseModel`` instance with a
  ``resilience_metadata`` field
- The response object has a ``resilience_metadata`` attribute

No ``print()`` calls are used anywhere in this module.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from typing import Any, Callable, Literal

from app.config.settings import get_settings
from app.exceptions.custom_exceptions import BaseApiException
from app.resilience.registry import resilience_registry
from app.utils.logger import LoggerFactory

_log = LoggerFactory.get_logger(__name__)

_PROFILE_TYPE = Literal["predict", "train", "eda"]


def resilience4j_pipeline(
    profile: _PROFILE_TYPE,
    fallback_callable: Callable[..., Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator factory that applies the full Resilience4j pipeline to a function.

    Parameters
    ----------
    profile:
        One of ``"predict"``, ``"train"``, or ``"eda"``.  Selects the set of
        Resilience4j component instances from the ``ResilienceRegistry``.
    fallback_callable:
        Optional callable invoked when the protected function raises any
        exception **and** the relevant ``ENABLE_FALLBACK_ON_*`` setting is
        ``True``.  It receives the original positional/keyword arguments plus
        two additional keyword arguments:
        - ``reason: str``            — the stringified exception
        - ``resilience_metadata: dict`` — the metadata built so far

    Returns
    -------
    Callable
        A decorator that wraps the target function with the Resilience4j chain.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            settings = get_settings()
            p = resilience_registry.get_profile(profile)

            # ------------------------------------------------------------------
            # Initial metadata template
            # ------------------------------------------------------------------
            metadata: dict[str, Any] = {
                "circuit_breaker_name": p.circuit_breaker.name,
                "circuit_breaker_state": p.circuit_breaker.state.value,
                "fallback_triggered": False,
                "fallback_reason": None,
                "retry_attempts_used": 0,
                "bulkhead_available_permits": p.bulkhead.get_metrics_snapshot()[
                    "available_concurrent_calls"
                ],
                "rate_limit_remaining": p.rate_limiter.get_metrics_snapshot()[
                    "available_permissions"
                ],
                "execution_time_ms": 0.0,
            }

            start_ns = time.perf_counter_ns()

            # ------------------------------------------------------------------
            # Layer 1: Fallback wrapper (outermost)
            # ------------------------------------------------------------------
            try:
                # Layer 2: CircuitBreaker
                p.circuit_breaker.allow_request()
                metadata["circuit_breaker_state"] = p.circuit_breaker.state.value

                try:
                    # Layer 3: RateLimiter
                    remaining_rate = p.rate_limiter.acquire_permission()
                    metadata["rate_limit_remaining"] = remaining_rate

                    # Layer 4 + 5 + 6: TimeLimiter wraps Bulkhead → Retry → Target
                    result, retries_used = await p.time_limiter.execute(
                        _bulkhead_retry_target,
                        p,
                        func,
                        args,
                        kwargs,
                        metadata,
                    )

                    metadata["retry_attempts_used"] = retries_used
                    metadata["bulkhead_available_permits"] = (
                        p.bulkhead.get_metrics_snapshot()["available_concurrent_calls"]
                    )

                    # Success path — record to circuit breaker
                    p.circuit_breaker.record_success()
                    metadata["circuit_breaker_state"] = p.circuit_breaker.state.value

                    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                    metadata["execution_time_ms"] = round(elapsed_ms, 3)

                    _log.info(
                        "resilience4j_pipeline [%s] '%s' succeeded in %.3fms "
                        "(retries=%d).",
                        profile,
                        getattr(func, "__name__", repr(func)),
                        elapsed_ms,
                        retries_used,
                    )

                    return _inject_metadata(result, metadata)

                except Exception as inner_exc:
                    # Record failure for non-resilience exceptions so the CB
                    # can track the failure rate.  Resilience4j control-flow
                    # exceptions (CB open, rate-limit, bulkhead, time-limit)
                    # are re-raised after recording.
                    p.circuit_breaker.record_failure(inner_exc)
                    metadata["circuit_breaker_state"] = p.circuit_breaker.state.value
                    raise

            except Exception as exc:
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                metadata["execution_time_ms"] = round(elapsed_ms, 3)

                # ------------------------------------------------------------------
                # Determine if fallback is enabled for this profile
                # ------------------------------------------------------------------
                fallback_enabled = False
                if profile == "predict":
                    fallback_enabled = settings.ENABLE_FALLBACK_ON_PREDICT
                elif profile == "train":
                    # No dedicated toggle — training fallback is not supported
                    fallback_enabled = False
                elif profile == "eda":
                    fallback_enabled = settings.ENABLE_FALLBACK_ON_METRICS

                if fallback_callable is not None and fallback_enabled:
                    metadata["fallback_triggered"] = True
                    metadata["fallback_reason"] = str(exc)
                    _log.warning(
                        "resilience4j_pipeline [%s] '%s' triggering fallback "
                        "after %.3fms due to: %s",
                        profile,
                        getattr(func, "__name__", repr(func)),
                        elapsed_ms,
                        exc,
                    )
                    try:
                        fallback_result = fallback_callable(
                            *args,
                            reason=str(exc),
                            resilience_metadata=metadata,
                            **kwargs,
                        )
                        # Support async fallbacks
                        if asyncio.iscoroutine(fallback_result):
                            fallback_result = await fallback_result
                        return _inject_metadata(fallback_result, metadata)
                    except Exception as fallback_exc:
                        _log.error(
                            "resilience4j_pipeline [%s] fallback itself failed: %s",
                            profile,
                            fallback_exc,
                        )
                        raise fallback_exc from exc

                # No fallback — re-raise original exception
                raise

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _bulkhead_retry_target(
    profile: Any,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[Any, int]:
    """
    Execute the Bulkhead → Retry → Target portion of the pipeline.

    This coroutine is passed to ``TimeLimiter.execute`` so that the combined
    Bulkhead acquisition + Retry loop is subject to the time budget.

    Parameters
    ----------
    profile:
        The ``ResilienceProfile`` for the current request.
    func:
        The original target function.
    args / kwargs:
        Original call arguments.
    metadata:
        Mutable metadata dict (updated with bulkhead state).

    Returns
    -------
    tuple[result, retries_used]
    """
    # Layer 5: Bulkhead
    await profile.bulkhead.acquire()
    try:
        metadata["bulkhead_available_permits"] = (
            profile.bulkhead.get_metrics_snapshot()["available_concurrent_calls"]
        )

        # Layer 6: Retry → Target function
        result, retries_used = await profile.retry.execute(func, *args, **kwargs)
        return result, retries_used
    finally:
        profile.bulkhead.release()


def _inject_metadata(result: Any, metadata: dict[str, Any]) -> Any:
    """
    Attempt to inject ``resilience_metadata`` into *result*.

    Injection rules (in priority order):
    1. If *result* is a ``dict``, add/update the ``"resilience_metadata"`` key.
    2. If *result* is a Pydantic ``BaseModel`` with a ``resilience_metadata``
       field, coerce the metadata dict to a ``ResilienceMetadataDTO`` instance
       and set the attribute via ``object.__setattr__`` (bypasses model
       validation freeze).
    3. If *result* has a ``resilience_metadata`` attribute (dataclass, custom
       object), set it.
    4. Otherwise return *result* unchanged.

    Parameters
    ----------
    result:
        The return value from the target function or fallback.
    metadata:
        The metadata dictionary to inject.

    Returns
    -------
    Any
        The (possibly modified) result.
    """
    if isinstance(result, dict):
        result["resilience_metadata"] = metadata
        return result

    # Pydantic v2 BaseModel check
    try:
        from pydantic import BaseModel  # noqa: PLC0415

        if isinstance(result, BaseModel) and "resilience_metadata" in type(result).model_fields:
            # Coerce the raw dict to a properly typed ResilienceMetadataDTO so
            # FastAPI's response serializer does not emit type-mismatch warnings.
            try:
                from app.dto.resilience_dto import ResilienceMetadataDTO  # noqa: PLC0415
                meta_dto = ResilienceMetadataDTO(**metadata)
                object.__setattr__(result, "resilience_metadata", meta_dto)
            except Exception:  # noqa: BLE001
                # Fallback: inject raw dict if DTO construction fails
                object.__setattr__(result, "resilience_metadata", metadata)
            return result
    except ImportError:
        pass

    # Generic attribute injection
    if hasattr(result, "resilience_metadata"):
        try:
            result.resilience_metadata = metadata
        except (AttributeError, TypeError):
            pass

    return result
