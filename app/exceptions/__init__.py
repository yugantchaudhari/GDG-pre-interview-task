# app/exceptions/__init__.py
# Re-exports the public surface of the exceptions package.
from app.exceptions.custom_exceptions import (
    BaseApiException,
    BulkheadFullException,
    CircuitBreakerOpenException,
    DatasetNotFoundException,
    InvalidDatasetSchemaException,
    InvalidModelOverrideException,
    ModelNotTrainedException,
    RateLimitExceededException,
    TimeLimiterTimeoutException,
)
from app.exceptions.global_handler import register_exception_handlers

__all__ = [
    "BaseApiException",
    "DatasetNotFoundException",
    "InvalidDatasetSchemaException",
    "ModelNotTrainedException",
    "InvalidModelOverrideException",
    "CircuitBreakerOpenException",
    "RateLimitExceededException",
    "BulkheadFullException",
    "TimeLimiterTimeoutException",
    "register_exception_handlers",
]
