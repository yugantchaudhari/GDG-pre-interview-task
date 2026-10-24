"""
app/dto/health_dto.py
======================
Pydantic v2 Data Transfer Object for the ``GET /api/v1/health`` endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponseDTO(BaseModel):
    """
    Aggregated health status of the Insurance Cost Predictor Microservice.

    Returned by ``GET /api/v1/health`` and includes dataset availability,
    model training state, and a full Resilience4j registry snapshot.
    """

    status: str = Field(
        default="UP",
        description="Overall service status ('UP' when all subsystems are healthy).",
    )
    dataset_available: bool = Field(
        ..., description="True if the configured dataset CSV exists and is non-empty."
    )
    dataset_path: str = Field(
        ..., description="Resolved filesystem path to the dataset that was checked."
    )
    model_trained: bool = Field(
        ...,
        description="True if at least one trained pipeline artefact is present on disk.",
    )
    active_best_model: str | None = Field(
        default=None,
        description="Name of the model selected as best during the last training run.",
    )
    last_trained_at: str | None = Field(
        default=None,
        description="ISO-8601 timestamp of the most recent completed training run.",
    )
    resilience4j_registry: dict = Field(
        ...,
        description=(
            "Nested snapshot of the full Resilience4j registry "
            "(predict / train / eda profiles with CB, RL, BH, TL, Retry metrics)."
        ),
    )
