# app/controllers/__init__.py
# Re-exports all router instances from the controllers package.

from app.controllers.eda_controller import router as eda_router
from app.controllers.health_controller import router as health_router
from app.controllers.prediction_controller import router as prediction_router
from app.controllers.training_controller import router as training_router

__all__ = [
    "health_router",
    "eda_router",
    "training_router",
    "prediction_router",
]
