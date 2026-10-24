"""
app/resilience/retry.py
========================
Native Python implementation of the Resilience4j Retry pattern with
exponential backoff.

Retry semantics
---------------
* Retries are only applied to **transient I/O or OS errors** (``OSError``,
  ``IOError``, ``MemoryError``).  These represent situations where the same
  request might succeed if retried moments later (e.g., a transient file-lock,
  a momentary memory allocation failure).
* Domain exceptions (``BaseApiException`` subclasses) and Resilience4j
  exceptions are **never retried** — they represent deliberate application
  decisions that should propagate immediately.
* The backoff formula is:
    ``wait = base_wait_sec * (backoff_multiplier ** attempt_index)``
  where ``attempt_index`` starts at 0 for the first retry (i.e., second
  attempt).

Return value
------------
The function returns a ``tuple[result, retries_used]`` so that the
``@resilience4j_pipeline`` decorator can record ``retry_attempts_used`` in
the resilience metadata without touching the result itself.

No ``print()`` calls are used anywhere in this module.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from app.exceptions.custom_exceptions import BaseApiException
from app.utils.logger import LoggerFactory

_log = LoggerFactory.get_logger(__name__)

#: Exception types that qualify for a retry attempt.
_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    OSError,
    IOError,
    MemoryError,
)


class Retry:
    """
    Exponential-backoff retry runner.

    Parameters
    ----------
    name:
        Human-readable identifier used in log messages.
    max_attempts:
        Total number of attempts (1 = no retries, 2 = one retry, etc.).
    wait_duration_sec:
        Base wait time in seconds between attempts.
    backoff_multiplier:
        Factor by which the wait time grows on each successive retry.
    """

    def __init__(
        self,
        name: str,
        max_attempts: int,
        wait_duration_sec: float,
        backoff_multiplier: float,
    ) -> None:
        self._name = name
        self._max_attempts = max(1, max_attempts)
        self._wait_duration_sec = wait_duration_sec
        self._backoff_multiplier = backoff_multiplier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The retry instance's human-readable name."""
        return self._name

    async def execute(self, func: Any, *args: Any, **kwargs: Any) -> tuple[Any, int]:
        """
        Execute *func* up to ``max_attempts`` times, retrying on transient errors.

        Parameters
        ----------
        func:
            Either an ``async def`` coroutine function or a regular synchronous
            callable.  Sync callables are awaited via ``asyncio.to_thread``.
        *args:
            Positional arguments forwarded to *func*.
        **kwargs:
            Keyword arguments forwarded to *func*.

        Returns
        -------
        tuple[Any, int]
            ``(result, retries_used)`` where ``retries_used`` is the number of
            retry attempts consumed (0 if the first attempt succeeded).

        Raises
        ------
        Exception
            The last exception raised after all attempts are exhausted, or any
            non-retryable exception on the first occurrence.
        """
        last_exception: Exception | None = None

        for attempt in range(self._max_attempts):
            try:
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = await asyncio.to_thread(func, *args, **kwargs)
                retries_used = attempt  # attempt=0 on first try, 1 on first retry…
                if attempt > 0:
                    _log.info(
                        "Retry '%s' succeeded on attempt %d/%d.",
                        self._name,
                        attempt + 1,
                        self._max_attempts,
                    )
                return result, retries_used

            except BaseApiException:
                # Domain / Resilience4j exceptions are not retried — propagate immediately.
                raise

            except _RETRYABLE_EXCEPTIONS as exc:  # type: ignore[misc]
                # pyrefly: ignore[bad-assignment]
                last_exception = exc
                remaining = self._max_attempts - attempt - 1
                if remaining <= 0:
                    _log.error(
                        "Retry '%s' exhausted all %d attempt(s). Last error: %s",
                        self._name,
                        self._max_attempts,
                        exc,
                    )
                    break

                wait_sec = self._wait_duration_sec * (
                    self._backoff_multiplier ** attempt
                )
                _log.warning(
                    "Retry '%s' attempt %d/%d failed with %s: %s — "
                    "retrying in %.3fs.",
                    self._name,
                    attempt + 1,
                    self._max_attempts,
                    type(exc).__name__,
                    exc,
                    wait_sec,
                )
                await asyncio.sleep(wait_sec)

            except Exception as exc:
                # Non-retryable, non-domain exception — propagate immediately.
                _log.error(
                    "Retry '%s' encountered non-retryable exception %s on attempt %d/%d: %s",
                    self._name,
                    type(exc).__name__,
                    attempt + 1,
                    self._max_attempts,
                    exc,
                )
                raise

        # All attempts exhausted — re-raise the last retryable exception.
        raise last_exception  # type: ignore[misc]
