"""
app/services/__init__.py
=========================
Service layer singleton instances and FastAPI ``Depends()``-compatible
getter functions.

Dependency injection pattern
----------------------------
All service classes are instantiated once at module load time as private
singletons.  Public getter functions return the same singleton on every call,
matching the FastAPI ``Depends(get_xxx_service)`` pattern.

Circular dependency note
------------------------
``TrainingService`` requires a reference to ``PredictionService`` for
post-training hot-reload.  The singletons are declared in dependency order:
PredictionService → TrainingService so there is no circular import.
"""

from __future__ import annotations

from app.services.dataset_service import DatasetService
from app.services.eda_service import EdaService
from app.services.fallback_service import FallbackService
from app.services.prediction_service import PredictionService
from app.services.preprocessing_service import PreprocessingService
from app.services.training_service import TrainingService

# ---------------------------------------------------------------------------
# Module-level singletons (instantiated in dependency order)
# ---------------------------------------------------------------------------

_dataset_service: DatasetService = DatasetService()
_preprocessing_service: PreprocessingService = PreprocessingService()
_eda_service: EdaService = EdaService(dataset_service=_dataset_service)
_prediction_service: PredictionService = PredictionService()
_training_service: TrainingService = TrainingService(
    dataset_service=_dataset_service,
    preprocessing_service=_preprocessing_service,
    prediction_service=_prediction_service,
)
_fallback_service: FallbackService = FallbackService()

# Eagerly attempt to load artefacts if they already exist from a previous run
_prediction_service.load_artifacts_into_memory()


# ---------------------------------------------------------------------------
# Public getter functions (FastAPI Depends-compatible)
# ---------------------------------------------------------------------------

def get_dataset_service() -> DatasetService:
    """Return the singleton DatasetService instance."""
    return _dataset_service


def get_eda_service() -> EdaService:
    """Return the singleton EdaService instance."""
    return _eda_service


def get_preprocessing_service() -> PreprocessingService:
    """Return the singleton PreprocessingService instance."""
    return _preprocessing_service


def get_prediction_service() -> PredictionService:
    """Return the singleton PredictionService instance."""
    return _prediction_service


def get_training_service() -> TrainingService:
    """Return the singleton TrainingService instance."""
    return _training_service


def get_fallback_service() -> FallbackService:
    """Return the singleton FallbackService instance."""
    return _fallback_service


__all__ = [
    "DatasetService",
    "EdaService",
    "PreprocessingService",
    "PredictionService",
    "TrainingService",
    "FallbackService",
    "get_dataset_service",
    "get_eda_service",
    "get_preprocessing_service",
    "get_prediction_service",
    "get_training_service",
    "get_fallback_service",
]
