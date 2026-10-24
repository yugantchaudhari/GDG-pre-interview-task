"""
tests/test_api.py
==================
Comprehensive Pytest suite for the Insurance Cost Predictor Microservice.

Tests cover all 6 functional and Resilience4j requirements:
1. Health endpoint correctness.
2. EDA summary endpoint correctness.
3. Model training + metrics round-trip.
4. Prediction endpoint — HIGH_RISK and LOW_RISK scenarios.
5. Prediction validation error (age below VALIDATION_AGE_MIN).
6. Resilience4j fallback when circuit breaker is forced OPEN.

All tests use FastAPI's ``TestClient`` (synchronous) backed by ``httpx``.
The TestClient triggers the ``lifespan`` startup hooks, which auto-trains
the model if no artefacts exist, so tests 4–6 run against a live model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.resilience.circuit_breaker import CircuitState
from app.resilience.registry import resilience_registry

# ---------------------------------------------------------------------------
# Module-level TestClient (shared across all tests in this module)
# ---------------------------------------------------------------------------
# Using a context manager ensures the lifespan (startup + shutdown) hooks run.
client = TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# Test 1: Health Endpoint
# ===========================================================================

def test_health_endpoint() -> None:
    """
    GET /api/v1/health must return HTTP 200 with:
    - status == "UP"
    - dataset_available is True
    - resilience4j_registry containing "predict", "train", and "eda" profiles
    """
    response = client.get("/api/v1/health")

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}. Body: {response.text}"
    )

    body = response.json()

    assert body["status"] == "UP", f"Expected status='UP', got {body['status']}"
    assert body["dataset_available"] is True, "Expected dataset_available=True"

    registry = body.get("resilience4j_registry", {})
    assert "predict" in registry, "Expected 'predict' profile in resilience4j_registry"
    assert "train" in registry, "Expected 'train' profile in resilience4j_registry"
    assert "eda" in registry, "Expected 'eda' profile in resilience4j_registry"


# ===========================================================================
# Test 2: EDA Summary Endpoint
# ===========================================================================

def test_eda_summary_endpoint() -> None:
    """
    GET /api/v1/eda/summary must return HTTP 200 with:
    - total_rows == 1338
    - duplicate_rows >= 1
    - charges flagged is_highly_skewed is True
    - smoker_cost_multiplier > 3.0
    - resilience_metadata.circuit_breaker_state == "CLOSED"
    """
    response = client.get("/api/v1/eda/summary")

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}. Body: {response.text}"
    )

    body = response.json()

    # Dataset shape assertions
    assert body["total_rows"] == 1338, (
        f"Expected 1338 rows, got {body['total_rows']}"
    )
    assert body["duplicate_rows"] >= 1, (
        f"Expected at least 1 duplicate row, got {body['duplicate_rows']}"
    )

    # Charges column should be highly skewed
    numerical_summary = body.get("numerical_summary", {})
    charges_stats = numerical_summary.get("charges", {})
    assert charges_stats.get("is_highly_skewed") is True, (
        "Expected charges.is_highly_skewed == True"
    )

    # Smoker cost multiplier
    domain_insights = body.get("domain_insights", {})
    smoker_multiplier = domain_insights.get("smoker_cost_multiplier", 0.0)
    assert smoker_multiplier > 3.0, (
        f"Expected smoker_cost_multiplier > 3.0, got {smoker_multiplier}"
    )

    # Resilience metadata
    meta = body.get("resilience_metadata")
    assert meta is not None, "Expected resilience_metadata to be present"
    assert meta["circuit_breaker_state"] == "CLOSED", (
        f"Expected CLOSED circuit breaker state, got {meta['circuit_breaker_state']}"
    )


# ===========================================================================
# Test 3: Model Train and Metrics Endpoints
# ===========================================================================

def test_model_train_and_metrics_endpoints() -> None:
    """
    POST /api/v1/model/train must return HTTP 200 with:
    - 3 models in leaderboard (LR, RF, GB)
    - best_model_selected has r2_mean > 0.80
    - Artifact files persisted to disk

    GET /api/v1/model/metrics must return HTTP 200 after training.
    """
    from app.config.settings import get_settings
    settings = get_settings()

    train_payload = {
        "cv_folds": 5,
        "drop_duplicates": True,
        "handle_feature_skewness": True,
        "apply_log_transform_on_target": True,
        "include_bmi_smoker_interaction": True,
    }

    train_response = client.post("/api/v1/model/train", json=train_payload)

    assert train_response.status_code == 200, (
        f"Expected 200 from /train, got {train_response.status_code}. Body: {train_response.text}"
    )

    body = train_response.json()

    # Must have exactly 3 models
    leaderboard = body.get("leaderboard", [])
    assert len(leaderboard) == 3, f"Expected 3 models in leaderboard, got {len(leaderboard)}"

    model_names_in_leaderboard = {entry["model_name"] for entry in leaderboard}
    assert "LinearRegression" in model_names_in_leaderboard, "LinearRegression must be in leaderboard"
    assert "RandomForestRegressor" in model_names_in_leaderboard, "RandomForestRegressor must be in leaderboard"
    assert "GradientBoostingRegressor" in model_names_in_leaderboard, "GradientBoostingRegressor must be in leaderboard"

    # Best model must have r2_mean > 0.80
    best_model_name = body.get("best_model_selected")
    best_entry = next(e for e in leaderboard if e["model_name"] == best_model_name)
    r2_mean = best_entry["metrics"]["r2_mean"]
    assert r2_mean > 0.80, (
        f"Expected best model R² > 0.80, got {r2_mean} for {best_model_name}"
    )

    # Artifact files must exist on disk
    assert settings.best_pipeline_path.exists(), (
        f"Best pipeline artefact not found at {settings.best_pipeline_path}"
    )
    assert settings.all_models_path.exists(), (
        f"All models artefact not found at {settings.all_models_path}"
    )
    assert settings.evaluation_report_path.exists(), (
        f"Evaluation report not found at {settings.evaluation_report_path}"
    )

    # Now test the metrics endpoint
    metrics_response = client.get("/api/v1/model/metrics")
    assert metrics_response.status_code == 200, (
        f"Expected 200 from /metrics, got {metrics_response.status_code}. Body: {metrics_response.text}"
    )


# ===========================================================================
# Test 4: Prediction — HIGH_RISK and LOW_RISK Scenarios
# ===========================================================================

def test_predict_endpoint_high_and_low_risk() -> None:
    """
    POST /api/v1/predict with a high-risk smoker+obese payload must return:
    - HTTP 200
    - predicted_charge > 15000
    - risk_assessment.risk_tier == "HIGH_RISK"
    - resilience_metadata.fallback_triggered is False

    POST /api/v1/predict with a low-risk non-smoker payload must return:
    - HTTP 200
    - risk_assessment.risk_tier == "LOW_RISK"
    """
    # --- High-risk scenario ---
    high_risk_payload = {
        "age": 34,
        "sex": "male",
        "bmi": 31.4,
        "children": 2,
        "smoker": "yes",
        "region": "southeast",
    }

    hr_response = client.post("/api/v1/predict", json=high_risk_payload)
    assert hr_response.status_code == 200, (
        f"Expected 200 for high-risk, got {hr_response.status_code}. Body: {hr_response.text}"
    )

    hr_body = hr_response.json()
    predicted_charge = hr_body.get("predicted_charge", 0.0)
    assert predicted_charge > 15000, (
        f"Expected predicted_charge > 15000 for high-risk smoker, got {predicted_charge}"
    )

    risk_tier = hr_body.get("risk_assessment", {}).get("risk_tier")
    assert risk_tier == "HIGH_RISK", (
        f"Expected HIGH_RISK for smoker, got {risk_tier}"
    )

    meta = hr_body.get("resilience_metadata")
    assert meta is not None, "Expected resilience_metadata in prediction response"
    assert meta["fallback_triggered"] is False, (
        f"Expected fallback_triggered=False for successful ML call, got {meta['fallback_triggered']}"
    )

    # --- Low-risk scenario ---
    low_risk_payload = {
        "age": 22,
        "sex": "female",
        "bmi": 21.5,
        "children": 0,
        "smoker": "no",
        "region": "northwest",
    }

    lr_response = client.post("/api/v1/predict", json=low_risk_payload)
    assert lr_response.status_code == 200, (
        f"Expected 200 for low-risk, got {lr_response.status_code}. Body: {lr_response.text}"
    )

    lr_body = lr_response.json()
    lr_risk_tier = lr_body.get("risk_assessment", {}).get("risk_tier")
    assert lr_risk_tier == "LOW_RISK", (
        f"Expected LOW_RISK for young non-smoker normal-BMI, got {lr_risk_tier}"
    )


# ===========================================================================
# Test 5: Prediction Validation Error
# ===========================================================================

def test_predict_validation_error() -> None:
    """
    POST /api/v1/predict with age=12 (below VALIDATION_AGE_MIN=18) must return:
    - HTTP 422
    - error_code == "VALIDATION_ERROR"
    """
    invalid_payload = {
        "age": 12,  # Below VALIDATION_AGE_MIN = 18
        "sex": "male",
        "bmi": 22.0,
        "children": 0,
        "smoker": "no",
        "region": "northeast",
    }

    response = client.post("/api/v1/predict", json=invalid_payload)
    assert response.status_code == 422, (
        f"Expected 422, got {response.status_code}. Body: {response.text}"
    )

    body = response.json()
    assert body.get("error_code") == "VALIDATION_ERROR", (
        f"Expected error_code='VALIDATION_ERROR', got {body.get('error_code')}"
    )


# ===========================================================================
# Test 6: Resilience4j Fallback When Circuit Breaker is OPEN
# ===========================================================================

def test_resilience4j_fallback_when_circuit_open() -> None:
    """
    Forcefully opens the 'predict' circuit breaker, then sends a valid predict
    request.  Expects HTTP 200 with:
    - model_used == "ActuarialHeuristicFallbackEngine"
    - resilience_metadata.fallback_triggered is True

    Resets circuit breaker to CLOSED after the test.
    """
    predict_profile = resilience_registry.get_profile("predict")
    cb = predict_profile.circuit_breaker

    # --- Force circuit breaker OPEN ---
    # Acquire the lock and set state directly to bypass the sliding window logic
    import time
    with cb._lock:  # noqa: SLF001
        cb._state = CircuitState.OPEN  # noqa: SLF001
        cb._opened_at = time.monotonic() + 9999  # far future — won't auto-recover  # noqa: SLF001

    try:
        valid_payload = {
            "age": 34,
            "sex": "male",
            "bmi": 31.4,
            "children": 2,
            "smoker": "yes",
            "region": "southeast",
        }

        response = client.post("/api/v1/predict", json=valid_payload)

        # When fallback is enabled (ENABLE_FALLBACK_ON_PREDICT=True) and CB is OPEN,
        # the decorator invokes the fallback and returns HTTP 200.
        assert response.status_code == 200, (
            f"Expected 200 with fallback, got {response.status_code}. Body: {response.text}"
        )

        body = response.json()

        assert body.get("model_used") == "ActuarialHeuristicFallbackEngine", (
            f"Expected ActuarialHeuristicFallbackEngine, got {body.get('model_used')}"
        )

        meta = body.get("resilience_metadata")
        assert meta is not None, "Expected resilience_metadata in fallback response"
        assert meta["fallback_triggered"] is True, (
            "Expected fallback_triggered=True when circuit is OPEN"
        )

    finally:
        # --- Reset circuit breaker back to CLOSED ---
        with cb._lock:  # noqa: SLF001
            cb._state = CircuitState.CLOSED  # noqa: SLF001
            cb._opened_at = 0.0  # noqa: SLF001
            cb._half_open_probe_count = 0  # noqa: SLF001
            cb._half_open_success_count = 0  # noqa: SLF001
            cb._window.clear()  # noqa: SLF001
