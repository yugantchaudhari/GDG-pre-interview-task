"""
app/utils/logger.py
===================
Spring Boot SLF4J-style console-only logging for the Insurance Cost Predictor
Microservice.

Design decisions
----------------
* **MDC (Mapped Diagnostic Context)** is emulated via Python's `contextvars`
  module.  Each asyncio Task (i.e. each HTTP request) gets its own `trace_id`
  stored in `trace_id_context` — a `ContextVar[str]`.  The `Slf4jMdcFilter`
  attaches the current value to every `LogRecord` so the format string can
  reference `%(trace_id)s`.

* **Console-only** output: a single `logging.StreamHandler` targeting
  `sys.stdout` is added exactly once per named logger.  No `FileHandler` or
  `RotatingFileHandler` is ever created.

* **Thread-safety**: `LoggerFactory._configured_loggers` is a module-level
  set protected by a `threading.Lock` so that concurrent startup threads do
  not double-configure the same logger.

* **No `print()` calls** anywhere in this module.
"""

from __future__ import annotations

import logging
import secrets
import sys
import threading
from contextvars import ContextVar

# ---------------------------------------------------------------------------
# MDC Trace ID context variable
# ---------------------------------------------------------------------------

#: ContextVar that holds an 8-character hex trace-id for the current
#: asyncio task / thread.  Defaults to "SYSTEM-STARTUP" during initialisation.
trace_id_context: ContextVar[str] = ContextVar("trace_id", default="SYSTEM-STARTUP")


# ---------------------------------------------------------------------------
# MDC filter — injects trace_id into every LogRecord
# ---------------------------------------------------------------------------

class Slf4jMdcFilter(logging.Filter):
    """
    A `logging.Filter` that reads the current `trace_id` from
    `trace_id_context` and attaches it as `record.trace_id`.

    This mirrors Spring Boot's MDC (Mapped Diagnostic Context) behaviour,
    enabling `[traceId=%(trace_id)s]` in the format string to render the
    per-request identifier automatically.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.trace_id = trace_id_context.get()
        return True


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

class LoggerFactory:
    """
    Singleton-style factory that creates and caches named loggers.

    Each logger is configured exactly once with:
    - A `Slf4jMdcFilter` for MDC trace-id injection.
    - A single `StreamHandler` targeting `sys.stdout`.
    - A `Formatter` using the format and date-format strings from `settings`.
    - Propagation disabled to prevent double-logging via the root logger.
    """

    _configured_loggers: set[str] = set()
    _lock: threading.Lock = threading.Lock()

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Return a fully configured `logging.Logger` for *name*.

        Parameters
        ----------
        name:
            Typically ``__name__`` of the calling module, matching the
            Spring Boot convention of per-class loggers.

        Returns
        -------
        logging.Logger
            A logger instance that writes SLF4J-formatted records to stdout
            with the current request's trace-id embedded in every line.
        """
        # Import here (not at module level) to avoid a circular import with
        # app.config.settings, which itself imports nothing from app.utils.
        from app.config.settings import get_settings  # noqa: PLC0415

        settings = get_settings()

        with LoggerFactory._lock:
            logger = logging.getLogger(name)

            if name in LoggerFactory._configured_loggers:
                return logger

            # Set level from settings
            numeric_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
            logger.setLevel(numeric_level)

            # Attach the MDC filter
            mdc_filter = Slf4jMdcFilter()
            logger.addFilter(mdc_filter)

            # Create the stdout handler
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(numeric_level)

            # Apply the SLF4J-style formatter from settings
            formatter = logging.Formatter(
                fmt=settings.LOG_FORMAT,
                datefmt=settings.LOG_DATE_FORMAT,
            )
            handler.setFormatter(formatter)
            handler.addFilter(mdc_filter)

            logger.addHandler(handler)

            # Prevent duplicate records from propagating to the root logger
            logger.propagate = False

            LoggerFactory._configured_loggers.add(name)

        return logger


# ---------------------------------------------------------------------------
# Public trace-id helper functions
# ---------------------------------------------------------------------------

def set_request_trace_id(custom_trace_id: str | None = None) -> str:
    """
    Set a new trace-id for the current asyncio task / thread.

    If *custom_trace_id* is provided (e.g. from an incoming ``X-Trace-Id``
    header) it is used verbatim (truncated or padded to 8 characters).
    Otherwise a cryptographically random 8-character hex token is generated.

    Parameters
    ----------
    custom_trace_id:
        Optional caller-supplied trace identifier.

    Returns
    -------
    str
        The 8-character trace-id that was set in the context.
    """
    if custom_trace_id:
        # Normalise to exactly 8 characters
        trace_id = custom_trace_id[:8].upper().ljust(8, "0")
    else:
        trace_id = secrets.token_hex(4).upper()  # 4 bytes → 8 hex chars

    trace_id_context.set(trace_id)
    return trace_id


def get_request_trace_id() -> str:
    """
    Return the trace-id currently stored in the context.

    Returns
    -------
    str
        The active 8-character trace-id, or ``"SYSTEM-STARTUP"`` if none has
        been set yet.
    """
    return trace_id_context.get()
