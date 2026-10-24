"""
app/resilience/circuit_breaker.py
==================================
Native Python implementation of the Resilience4j CircuitBreaker pattern.

State machine
-------------
    CLOSED ──(failure rate >= threshold after min_calls)──> OPEN
    OPEN   ──(wait_duration elapsed)──> HALF_OPEN
    HALF_OPEN ──(probe calls all succeed)──> CLOSED
    HALF_OPEN ──(any probe call fails)──> OPEN

The sliding window is a fixed-size ``collections.deque`` of boolean outcomes
(``True`` = success, ``False`` = failure).  The failure rate is computed as
``failed_calls / total_calls_in_window * 100`` and compared against
``failure_rate_threshold`` once ``min_number_of_calls`` have been buffered.

No ``print()`` calls are used anywhere in this module.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from enum import Enum
from typing import Any

from app.exceptions.custom_exceptions import CircuitBreakerOpenException
from app.utils.logger import LoggerFactory

_log = LoggerFactory.get_logger(__name__)


# ---------------------------------------------------------------------------
# State enumeration
# ---------------------------------------------------------------------------

class CircuitState(str, Enum):
    """Enumeration of the three circuit-breaker states."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


# ---------------------------------------------------------------------------
# CircuitBreaker implementation
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Resilience4j-compatible circuit breaker implemented natively in Python.

    Parameters
    ----------
    name:
        Human-readable identifier used in log messages and metrics snapshots.
    failure_rate_threshold:
        Percentage (0–100) of failures that trips the breaker (e.g. ``50.0``).
    sliding_window_size:
        Maximum number of call outcomes tracked in the sliding window.
    min_number_of_calls:
        Minimum calls that must be in the window before the failure rate is
        evaluated.  Prevents tripping on a single cold-start failure.
    wait_duration_in_open_state_sec:
        Seconds the breaker remains ``OPEN`` before transitioning to
        ``HALF_OPEN``.
    permitted_calls_in_half_open:
        Number of probe calls allowed in ``HALF_OPEN`` state.  If all succeed
        the breaker transitions to ``CLOSED``; if any fail it returns to
        ``OPEN``.
    """

    def __init__(
        self,
        name: str,
        failure_rate_threshold: float,
        sliding_window_size: int,
        min_number_of_calls: int,
        wait_duration_in_open_state_sec: float,
        permitted_calls_in_half_open: int,
    ) -> None:
        self._name = name
        self._failure_rate_threshold = failure_rate_threshold
        self._sliding_window_size = sliding_window_size
        self._min_number_of_calls = min_number_of_calls
        self._wait_duration_in_open_state_sec = wait_duration_in_open_state_sec
        self._permitted_calls_in_half_open = permitted_calls_in_half_open

        self._state: CircuitState = CircuitState.CLOSED
        # Deque stores True (success) / False (failure) per call
        self._window: deque[bool] = deque(maxlen=sliding_window_size)
        self._opened_at: float = 0.0
        self._half_open_probe_count: int = 0
        self._half_open_success_count: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The circuit breaker's human-readable name."""
        return self._name

    @property
    def state(self) -> CircuitState:
        """Current state of the circuit breaker (thread-safe snapshot)."""
        with self._lock:
            return self._state

    def allow_request(self) -> bool:
        """
        Determine whether the current request is permitted to proceed.

        Returns
        -------
        bool
            ``True`` when the circuit is ``CLOSED`` or a probe is permitted in
            ``HALF_OPEN`` state.

        Raises
        ------
        CircuitBreakerOpenException
            When the circuit is ``OPEN`` and the wait duration has **not** yet
            elapsed, or when the ``HALF_OPEN`` probe quota is exhausted.
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self._wait_duration_in_open_state_sec:
                    self._transition_to_half_open()
                    return True
                raise CircuitBreakerOpenException(
                    instance_name=self._name,
                    details={
                        "state": self._state.value,
                        "retry_after_seconds": round(
                            self._wait_duration_in_open_state_sec - elapsed, 3
                        ),
                    },
                )

            # HALF_OPEN
            if self._half_open_probe_count < self._permitted_calls_in_half_open:
                self._half_open_probe_count += 1
                return True

            raise CircuitBreakerOpenException(
                instance_name=self._name,
                message=(
                    f"Circuit breaker '{self._name}' is HALF_OPEN but probe quota "
                    f"({self._permitted_calls_in_half_open}) is exhausted."
                ),
                details={"state": self._state.value},
            )

    def record_success(self) -> None:
        """
        Register a successful call outcome.

        In ``HALF_OPEN`` state, if all permitted probe calls have succeeded the
        breaker transitions back to ``CLOSED``.
        """
        with self._lock:
            self._window.append(True)
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_success_count += 1
                if self._half_open_success_count >= self._permitted_calls_in_half_open:
                    self._transition_to_closed()

    def record_failure(self, exc: Exception) -> None:
        """
        Register a failed call outcome and evaluate the failure rate.

        If the breaker is in ``HALF_OPEN`` it transitions immediately to
        ``OPEN`` on the first failure.  If it is ``CLOSED`` and the failure
        rate exceeds the threshold after ``min_number_of_calls`` it trips to
        ``OPEN``.

        Parameters
        ----------
        exc:
            The exception that caused the failure (used for log context).
        """
        with self._lock:
            self._window.append(False)

            if self._state == CircuitState.HALF_OPEN:
                _log.warning(
                    "CircuitBreaker '%s' probe call failed in HALF_OPEN — "
                    "returning to OPEN. Cause: %s",
                    self._name,
                    exc,
                )
                self._transition_to_open()
                return

            if self._state == CircuitState.CLOSED:
                self._evaluate_failure_rate()

    def get_metrics_snapshot(self) -> dict[str, Any]:
        """
        Return a point-in-time snapshot of the circuit breaker's metrics.

        Intended for the ``/api/v1/health`` endpoint.

        Returns
        -------
        dict
            ``state``, ``failure_rate_percent``, ``buffered_calls``,
            ``failed_calls``, ``wait_duration_open_state_sec``.
        """
        with self._lock:
            total = len(self._window)
            failed = self._window.count(False)
            failure_rate = (failed / total * 100.0) if total > 0 else 0.0
            return {
                "name": self._name,
                "state": self._state.value,
                "failure_rate_percent": round(failure_rate, 2),
                "buffered_calls": total,
                "failed_calls": failed,
                "wait_duration_open_state_sec": self._wait_duration_in_open_state_sec,
                "permitted_calls_in_half_open": self._permitted_calls_in_half_open,
            }

    # ------------------------------------------------------------------
    # Private state-transition helpers (must be called with _lock held)
    # ------------------------------------------------------------------

    def _evaluate_failure_rate(self) -> None:
        """Trip to OPEN if failure rate threshold is breached."""
        total = len(self._window)
        if total < self._min_number_of_calls:
            return
        failed = self._window.count(False)
        failure_rate = failed / total * 100.0
        if failure_rate >= self._failure_rate_threshold:
            _log.error(
                "CircuitBreaker '%s' CLOSED → OPEN: failure rate %.1f%% >= threshold %.1f%% "
                "(%d/%d calls failed in sliding window).",
                self._name,
                failure_rate,
                self._failure_rate_threshold,
                failed,
                total,
            )
            self._transition_to_open()

    def _transition_to_open(self) -> None:
        """Unconditionally move to OPEN state."""
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._half_open_probe_count = 0
        self._half_open_success_count = 0

    def _transition_to_half_open(self) -> None:
        """Move from OPEN to HALF_OPEN and reset probe counters."""
        _log.info(
            "CircuitBreaker '%s' OPEN → HALF_OPEN: "
            "wait duration (%.1fs) elapsed — allowing %d probe call(s).",
            self._name,
            self._wait_duration_in_open_state_sec,
            self._permitted_calls_in_half_open,
        )
        self._state = CircuitState.HALF_OPEN
        self._half_open_probe_count = 0
        self._half_open_success_count = 0

    def _transition_to_closed(self) -> None:
        """Move from HALF_OPEN to CLOSED and clear the sliding window."""
        _log.info(
            "CircuitBreaker '%s' HALF_OPEN → CLOSED: "
            "all %d probe call(s) succeeded.",
            self._name,
            self._permitted_calls_in_half_open,
        )
        self._state = CircuitState.CLOSED
        self._window.clear()
        self._half_open_probe_count = 0
        self._half_open_success_count = 0
