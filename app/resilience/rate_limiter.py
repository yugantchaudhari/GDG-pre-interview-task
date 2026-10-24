"""
app/resilience/rate_limiter.py
================================
Native Python implementation of the Resilience4j RateLimiter pattern.

Algorithm: **sliding-window** based on monotonic timestamps.
Each call that arrives records ``time.monotonic()`` in a ``deque``.  Before
evaluating capacity, all timestamps older than ``refresh_period_sec`` are
evicted.  If the number of remaining timestamps equals or exceeds
``limit_for_period`` the call is rejected with ``RateLimitExceededException``.

This approach naturally handles bursts at window boundaries without the
per-period hard-reset artefacts of a fixed-window counter.

No ``print()`` calls are used anywhere in this module.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from app.exceptions.custom_exceptions import RateLimitExceededException
from app.utils.logger import LoggerFactory

_log = LoggerFactory.get_logger(__name__)


class RateLimiter:
    """
    Sliding-window rate limiter.

    Parameters
    ----------
    name:
        Human-readable identifier used in log messages and metrics snapshots.
    limit_for_period:
        Maximum number of calls permitted within any ``refresh_period_sec``
        window (e.g. ``60`` calls per minute).
    refresh_period_sec:
        The width of the sliding time window in seconds.
    """

    def __init__(
        self,
        name: str,
        limit_for_period: int,
        refresh_period_sec: float,
    ) -> None:
        self._name = name
        self._limit_for_period = limit_for_period
        self._refresh_period_sec = refresh_period_sec
        # Deque of monotonic timestamps for calls within the current window
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The rate limiter's human-readable name."""
        return self._name

    def acquire_permission(self) -> int:
        """
        Attempt to acquire a call permit for the current sliding window.

        Side-effects
        ------------
        * Purges timestamps older than ``refresh_period_sec`` from the deque.
        * Appends the current monotonic time to the deque on success.

        Returns
        -------
        int
            The number of remaining permissions in the current window
            (``limit_for_period - current_count``).

        Raises
        ------
        RateLimitExceededException
            When the window is already at or above ``limit_for_period``,
            including a ``retry_after_seconds`` value indicating how long
            until the oldest call exits the window.
        """
        with self._lock:
            now = time.monotonic()
            self._evict_old_timestamps(now)

            current_count = len(self._timestamps)
            if current_count >= self._limit_for_period:
                # The oldest timestamp is the earliest that will exit the window
                oldest = self._timestamps[0]
                retry_after = max(0.0, self._refresh_period_sec - (now - oldest))
                _log.warning(
                    "RateLimiter '%s' exceeded: %d/%d calls in current window. "
                    "Retry after %.3fs.",
                    self._name,
                    current_count,
                    self._limit_for_period,
                    retry_after,
                )
                raise RateLimitExceededException(
                    instance_name=self._name,
                    retry_after_seconds=round(retry_after, 3),
                    details={
                        "current_count": current_count,
                        "limit_for_period": self._limit_for_period,
                        "refresh_period_sec": self._refresh_period_sec,
                    },
                )

            self._timestamps.append(now)
            remaining = self._limit_for_period - len(self._timestamps)
            _log.debug(
                "RateLimiter '%s' permit acquired — %d/%d used, %d remaining.",
                self._name,
                len(self._timestamps),
                self._limit_for_period,
                remaining,
            )
            return remaining

    def get_metrics_snapshot(self) -> dict[str, Any]:
        """
        Return a point-in-time snapshot of the rate limiter's metrics.

        Returns
        -------
        dict
            ``limit_for_period``, ``limit_refresh_period_seconds``, and
            ``available_permissions``.
        """
        with self._lock:
            now = time.monotonic()
            self._evict_old_timestamps(now)
            available = max(0, self._limit_for_period - len(self._timestamps))
            return {
                "name": self._name,
                "limit_for_period": self._limit_for_period,
                "limit_refresh_period_seconds": self._refresh_period_sec,
                "available_permissions": available,
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_old_timestamps(self, now: float) -> None:
        """Remove timestamps that have fallen outside the current window."""
        cutoff = now - self._refresh_period_sec
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()
