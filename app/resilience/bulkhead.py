"""
app/resilience/bulkhead.py
===========================
Native Python implementation of the Resilience4j Bulkhead pattern.

The Bulkhead limits the number of concurrent calls to a protected service
using ``asyncio.Semaphore``.  A caller that cannot acquire a permit within
``max_wait_duration_sec`` receives a ``BulkheadFullException`` (HTTP 429)
rather than being queued indefinitely.

When ``max_wait_duration_sec == 0.0`` the acquire is attempted immediately
(non-blocking); any contention is rejected instantly — matching the
Resilience4j ``SemaphoreBulkhead`` behaviour when
``maxWaitDuration = Duration.ZERO``.

No ``print()`` calls are used anywhere in this module.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.exceptions.custom_exceptions import BulkheadFullException
from app.utils.logger import LoggerFactory

_log = LoggerFactory.get_logger(__name__)


class Bulkhead:
    """
    Asyncio-based semaphore bulkhead.

    Parameters
    ----------
    name:
        Human-readable identifier used in log messages and metrics snapshots.
    max_concurrent_calls:
        Maximum number of calls that may execute concurrently.
    max_wait_duration_sec:
        Seconds a caller will wait for a permit before raising
        ``BulkheadFullException``.  Use ``0.0`` for fail-fast (no wait).
    """

    def __init__(
        self,
        name: str,
        max_concurrent_calls: int,
        max_wait_duration_sec: float,
    ) -> None:
        self._name = name
        self._max_concurrent_calls = max_concurrent_calls
        self._max_wait_duration_sec = max_wait_duration_sec
        self._semaphore = asyncio.Semaphore(max_concurrent_calls)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The bulkhead's human-readable name."""
        return self._name

    async def acquire(self) -> None:
        """
        Acquire a concurrency permit.

        Waits up to ``max_wait_duration_sec`` for the semaphore.  If the wait
        times out (or there is no wait and the semaphore is exhausted) a
        ``BulkheadFullException`` is raised.

        Raises
        ------
        BulkheadFullException
            When no permit could be obtained within the allowed wait duration.
        """
        try:
            if self._max_wait_duration_sec <= 0.0:
                # Non-blocking: attempt a single acquire with a tiny timeout
                await asyncio.wait_for(self._semaphore.acquire(), timeout=1e-6)
            else:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self._max_wait_duration_sec,
                )
            _log.debug(
                "Bulkhead '%s' permit acquired — %d/%d permits in use.",
                self._name,
                self._max_concurrent_calls - self._semaphore._value,  # noqa: SLF001
                self._max_concurrent_calls,
            )
        except asyncio.TimeoutError:
            _log.warning(
                "Bulkhead '%s' is full — could not acquire permit within %.3fs. "
                "Current semaphore value: %d.",
                self._name,
                self._max_wait_duration_sec,
                self._semaphore._value,  # noqa: SLF001
            )
            raise BulkheadFullException(
                instance_name=self._name,
                details={
                    "max_concurrent_calls": self._max_concurrent_calls,
                    "max_wait_duration_sec": self._max_wait_duration_sec,
                },
            )

    def release(self) -> None:
        """
        Release a previously acquired concurrency permit back to the pool.

        Must be called in a ``finally`` block paired with every successful
        ``await acquire()`` call.
        """
        self._semaphore.release()
        _log.debug(
            "Bulkhead '%s' permit released — %d/%d permits now available.",
            self._name,
            self._semaphore._value,  # noqa: SLF001
            self._max_concurrent_calls,
        )

    def get_metrics_snapshot(self) -> dict[str, Any]:
        """
        Return a point-in-time snapshot of the bulkhead's metrics.

        Returns
        -------
        dict
            ``max_concurrent_calls``, ``available_concurrent_calls``,
            ``max_wait_duration_sec``.
        """
        available = self._semaphore._value  # noqa: SLF001
        return {
            "name": self._name,
            "max_concurrent_calls": self._max_concurrent_calls,
            "available_concurrent_calls": available,
            "max_wait_duration_sec": self._max_wait_duration_sec,
        }
