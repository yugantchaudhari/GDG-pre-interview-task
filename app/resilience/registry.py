"""
app/resilience/registry.py
============================
Singleton registry that holds all named Resilience4j component instances.

The ``ResilienceRegistry`` is initialised once from ``get_settings()`` and
creates three named profiles — ``"predict"``, ``"train"``, and ``"eda"`` —
each containing a ``CircuitBreaker``, ``RateLimiter``, ``Bulkhead``,
``TimeLimiter``, and ``Retry`` configured from the corresponding ``Settings``
fields.

The module-level singleton ``resilience_registry`` is the canonical entry
point for all decorators and health-check code.

No ``print()`` calls are used anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config.settings import get_settings
from app.resilience.bulkhead import Bulkhead
from app.resilience.circuit_breaker import CircuitBreaker
from app.resilience.rate_limiter import RateLimiter
from app.resilience.retry import Retry
from app.resilience.time_limiter import TimeLimiter
from app.utils.logger import LoggerFactory

_log = LoggerFactory.get_logger(__name__)


@dataclass
class ResilienceProfile:
    """
    A named bundle of all Resilience4j instances for a single endpoint profile.

    Attributes
    ----------
    circuit_breaker:
        The ``CircuitBreaker`` instance for this profile.
    rate_limiter:
        The ``RateLimiter`` instance for this profile.
    bulkhead:
        The ``Bulkhead`` instance for this profile.
    time_limiter:
        The ``TimeLimiter`` instance for this profile.
    retry:
        The ``Retry`` instance for this profile.
    """

    circuit_breaker: CircuitBreaker
    rate_limiter: RateLimiter
    bulkhead: Bulkhead
    time_limiter: TimeLimiter
    retry: Retry


class ResilienceRegistry:
    """
    Singleton that owns all ``ResilienceProfile`` instances.

    Access individual profiles via ``get_profile(name)`` and retrieve a full
    health snapshot via ``get_full_registry_snapshot()``.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._profiles: dict[str, ResilienceProfile] = {}
        self._build_predict_profile(settings)
        self._build_train_profile(settings)
        self._build_eda_profile(settings)
        _log.info(
            "ResilienceRegistry initialised with profiles: %s",
            list(self._profiles.keys()),
        )

    # ------------------------------------------------------------------
    # Profile builders
    # ------------------------------------------------------------------

    def _build_predict_profile(self, s: Any) -> None:
        self._profiles["predict"] = ResilienceProfile(
            circuit_breaker=CircuitBreaker(
                name="predict-cb",
                failure_rate_threshold=s.CB_PREDICT_FAILURE_RATE_THRESHOLD,
                sliding_window_size=s.CB_PREDICT_SLIDING_WINDOW_SIZE,
                min_number_of_calls=s.CB_PREDICT_MIN_NUMBER_OF_CALLS,
                wait_duration_in_open_state_sec=s.CB_PREDICT_WAIT_DURATION_IN_OPEN_STATE_SEC,
                permitted_calls_in_half_open=s.CB_PREDICT_PERMITTED_CALLS_IN_HALF_OPEN,
            ),
            rate_limiter=RateLimiter(
                name="predict-rl",
                limit_for_period=s.RL_PREDICT_LIMIT_FOR_PERIOD,
                refresh_period_sec=s.RL_PREDICT_REFRESH_PERIOD_SEC,
            ),
            bulkhead=Bulkhead(
                name="predict-bh",
                max_concurrent_calls=s.BH_PREDICT_MAX_CONCURRENT_CALLS,
                max_wait_duration_sec=s.BH_PREDICT_MAX_WAIT_DURATION_SEC,
            ),
            time_limiter=TimeLimiter(
                name="predict-tl",
                timeout_sec=s.TL_PREDICT_TIMEOUT_SEC,
            ),
            retry=Retry(
                name="predict-retry",
                max_attempts=s.RETRY_MAX_ATTEMPTS,
                wait_duration_sec=s.RETRY_WAIT_DURATION_SEC,
                backoff_multiplier=s.RETRY_BACKOFF_MULTIPLIER,
            ),
        )

    def _build_train_profile(self, s: Any) -> None:
        self._profiles["train"] = ResilienceProfile(
            circuit_breaker=CircuitBreaker(
                name="train-cb",
                failure_rate_threshold=s.CB_TRAIN_FAILURE_RATE_THRESHOLD,
                sliding_window_size=s.CB_TRAIN_SLIDING_WINDOW_SIZE,
                min_number_of_calls=s.CB_TRAIN_MIN_NUMBER_OF_CALLS,
                wait_duration_in_open_state_sec=s.CB_TRAIN_WAIT_DURATION_IN_OPEN_STATE_SEC,
                permitted_calls_in_half_open=s.CB_TRAIN_PERMITTED_CALLS_IN_HALF_OPEN,
            ),
            rate_limiter=RateLimiter(
                name="train-rl",
                limit_for_period=s.RL_TRAIN_LIMIT_FOR_PERIOD,
                refresh_period_sec=s.RL_TRAIN_REFRESH_PERIOD_SEC,
            ),
            bulkhead=Bulkhead(
                name="train-bh",
                max_concurrent_calls=s.BH_TRAIN_MAX_CONCURRENT_CALLS,
                max_wait_duration_sec=s.BH_TRAIN_MAX_WAIT_DURATION_SEC,
            ),
            time_limiter=TimeLimiter(
                name="train-tl",
                timeout_sec=s.TL_TRAIN_TIMEOUT_SEC,
            ),
            retry=Retry(
                name="train-retry",
                max_attempts=s.RETRY_MAX_ATTEMPTS,
                wait_duration_sec=s.RETRY_WAIT_DURATION_SEC,
                backoff_multiplier=s.RETRY_BACKOFF_MULTIPLIER,
            ),
        )

    def _build_eda_profile(self, s: Any) -> None:
        self._profiles["eda"] = ResilienceProfile(
            circuit_breaker=CircuitBreaker(
                name="eda-cb",
                failure_rate_threshold=s.CB_EDA_FAILURE_RATE_THRESHOLD,
                sliding_window_size=s.CB_EDA_SLIDING_WINDOW_SIZE,
                min_number_of_calls=s.CB_EDA_MIN_NUMBER_OF_CALLS,
                wait_duration_in_open_state_sec=s.CB_EDA_WAIT_DURATION_IN_OPEN_STATE_SEC,
                permitted_calls_in_half_open=s.CB_EDA_PERMITTED_CALLS_IN_HALF_OPEN,
            ),
            rate_limiter=RateLimiter(
                name="eda-rl",
                limit_for_period=s.RL_EDA_LIMIT_FOR_PERIOD,
                refresh_period_sec=s.RL_EDA_REFRESH_PERIOD_SEC,
            ),
            bulkhead=Bulkhead(
                name="eda-bh",
                max_concurrent_calls=s.BH_EDA_MAX_CONCURRENT_CALLS,
                max_wait_duration_sec=s.BH_EDA_MAX_WAIT_DURATION_SEC,
            ),
            time_limiter=TimeLimiter(
                name="eda-tl",
                timeout_sec=s.TL_EDA_TIMEOUT_SEC,
            ),
            retry=Retry(
                name="eda-retry",
                max_attempts=s.RETRY_MAX_ATTEMPTS,
                wait_duration_sec=s.RETRY_WAIT_DURATION_SEC,
                backoff_multiplier=s.RETRY_BACKOFF_MULTIPLIER,
            ),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_profile(self, name: str) -> ResilienceProfile:
        """
        Retrieve the ``ResilienceProfile`` for the given endpoint name.

        Parameters
        ----------
        name:
            One of ``"predict"``, ``"train"``, or ``"eda"``.

        Returns
        -------
        ResilienceProfile
            The corresponding profile bundle.

        Raises
        ------
        KeyError
            If *name* is not a registered profile.
        """
        try:
            return self._profiles[name]
        except KeyError:
            available = list(self._profiles.keys())
            raise KeyError(
                f"ResilienceRegistry: profile '{name}' not found. "
                f"Available profiles: {available}"
            ) from None

    def get_full_registry_snapshot(self) -> dict[str, Any]:
        """
        Return a nested snapshot of every profile's component metrics.

        Used by the ``/api/v1/health`` endpoint to surface live Resilience4j
        state without coupling the health controller to individual components.

        Returns
        -------
        dict
            Mapping of profile name → per-component metric dictionaries.
        """
        snapshot: dict[str, Any] = {}
        for profile_name, profile in self._profiles.items():
            snapshot[profile_name] = {
                "circuit_breaker": profile.circuit_breaker.get_metrics_snapshot(),
                "rate_limiter": profile.rate_limiter.get_metrics_snapshot(),
                "bulkhead": profile.bulkhead.get_metrics_snapshot(),
                "time_limiter": {
                    "name": profile.time_limiter.name,
                    "timeout_sec": profile.time_limiter.timeout_sec,
                },
                "retry": {
                    "name": profile.retry.name,
                    "max_attempts": profile.retry._max_attempts,  # noqa: SLF001
                    "wait_duration_sec": profile.retry._wait_duration_sec,  # noqa: SLF001
                    "backoff_multiplier": profile.retry._backoff_multiplier,  # noqa: SLF001
                },
            }
        return snapshot


# ---------------------------------------------------------------------------
# Module-level singleton — imported by decorators and health controllers
# ---------------------------------------------------------------------------

resilience_registry: ResilienceRegistry = ResilienceRegistry()
