"""
app/resilience/time_limiter.py
================================
Native Python implementation of the Resilience4j TimeLimiter pattern.

The TimeLimiter guards a coroutine (or a synchronous CPU/ML function) against
exceeding a wall-clock time budget.

* **Async functions** are awaited directly inside ``asyncio.wait_for``.
* **Sync functions** are dispatched to a thread pool via ``asyncio.to_thread``
  so that the event loop is not blocked, then guarded by ``asyncio.wait_for``.

On timeout, ``asyncio.TimeoutError`` is caught, logged as an error, and
re-raised as a domain-typed ``TimeLimiterTimeoutException`` (HTTP 504).

No ``print()`` calls are used anywhere in this module.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from app.exceptions.custom_exceptions import TimeLimiterTimeoutException
from app.utils.logger import LoggerFactory

_log = LoggerFactory.get_logger(__name__)


class TimeLimiter:
    """
    Asyncio-based time limiter that enforces a maximum execution duration.

    Parameters
    ----------
    name:
        Human-readable identifier used in log messages.
    timeout_sec:
        Maximum wall-clock seconds the protected callable is allowed to run.
    """

    def __init__(self, name: str, timeout_sec: float) -> None:
        self._name = name
        self._timeout_sec = timeout_sec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The time limiter's human-readable name."""
        return self._name

    @property
    def timeout_sec(self) -> float:
        """Configured timeout in seconds."""
        return self._timeout_sec

    async def execute(self, coro_func: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Execute *coro_func* with a time limit of ``self.timeout_sec``.

        Parameters
        ----------
        coro_func:
            Either an ``async def`` coroutine function or a regular synchronous
            callable.  Sync callables are run in a thread pool via
            ``asyncio.to_thread`` to keep the event loop non-blocking.
        *args:
            Positional arguments forwarded to *coro_func*.
        **kwargs:
            Keyword arguments forwarded to *coro_func*.

        Returns
        -------
        Any
            The return value of *coro_func*.

        Raises
        ------
        TimeLimiterTimeoutException
            When *coro_func* does not complete within ``timeout_sec``.
        """
        _log.debug(
            "TimeLimiter '%s' executing '%s' with timeout=%.3fs.",
            self._name,
            getattr(coro_func, "__name__", repr(coro_func)),
            self._timeout_sec,
        )

        try:
            if inspect.iscoroutinefunction(coro_func):
                coro = coro_func(*args, **kwargs)
            else:
                # Run synchronous CPU/ML function in a thread pool so the
                # event loop remains unblocked for other requests.
                coro = asyncio.to_thread(coro_func, *args, **kwargs)

            return await asyncio.wait_for(coro, timeout=self._timeout_sec)

        except asyncio.TimeoutError:
            func_name = getattr(coro_func, "__name__", repr(coro_func))
            _log.error(
                "TimeLimiter '%s' timed out after %.3fs while executing '%s'.",
                self._name,
                self._timeout_sec,
                func_name,
            )
            raise TimeLimiterTimeoutException(
                instance_name=self._name,
                timeout_sec=self._timeout_sec,
                details={"protected_function": func_name},
            )
