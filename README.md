# Insurance Cost Predictor Microservice

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-orange.svg)](https://scikit-learn.org)
[![Resilience4j](https://img.shields.io/badge/Resilience4j-Native%20Python-red.svg)](#fault-tolerance)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Enterprise-grade insurance cost prediction microservice** built with FastAPI, scikit-learn, and a native Python Resilience4j fault-tolerance pipeline — featuring Circuit Breaker, Rate Limiter, Bulkhead, Time Limiter, Retry, and an Actuarial Heuristic Fallback Engine.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Enterprise Architecture](#enterprise-architecture)
3. [Core ML Tasks](#core-ml-tasks)
4. [Fault-Tolerance Matrix](#fault-tolerance-matrix)
5. [API Reference](#api-reference)
6. [Quickstart](#quickstart)
7. [Example Requests & Responses](#example-requests--responses)
8. [Configuration Reference](#configuration-reference)

---

## Project Overview

The **Insurance Cost Predictor Microservice** solves the US health insurance cost estimation problem using a fully automated ML pipeline that:

- **Explores** the Kaggle Insurance dataset (1,338 records × 7 features).
- **Preprocesses** raw data with Yeo-Johnson power transforms, OneHotEncoding, and domain interaction features.
- **Trains and compares** three scikit-learn regressors via 5-fold cross-validation.
- **Predicts** real-time insurance charges with actuarial risk profiling.
- **Degrades gracefully** under failure via an Actuarial Heuristic Fallback Engine.

All configuration (thresholds, hyperparameters, resilience parameters) is loaded exclusively from `.env` — **zero hardcoding** in any source file.

---

## Enterprise Architecture

```
D:\pree innterview taks GDG\
│
├── app/
│   ├── config/
│   │   └── settings.py          ← Pydantic BaseSettings (12 groups, .env-driven)
│   │
│   ├── utils/
│   │   └── logger.py            ← SLF4J-style MDC logger (contextvars trace IDs)
│   │
│   ├── exceptions/
│   │   ├── custom_exceptions.py ← BaseApiException hierarchy (domain + Resilience4j)
│   │   └── global_handler.py    ← FastAPI register_exception_handlers()
│   │
│   ├── resilience/
│   │   ├── circuit_breaker.py   ← CircuitBreaker (CLOSED → OPEN → HALF_OPEN)
│   │   ├── rate_limiter.py      ← Token-bucket RateLimiter
│   │   ├── bulkhead.py          ← Semaphore-based Bulkhead
│   │   ├── time_limiter.py      ← asyncio.wait_for TimeLimiter
│   │   ├── retry.py             ← Exponential backoff Retry
│   │   ├── registry.py          ← ResilienceRegistry (predict / train / eda profiles)
│   │   └── decorators.py        ← @resilience4j_pipeline (master decorator)
│   │
│   ├── dto/
│   │   ├── resilience_dto.py    ← ErrorResponseDTO, ResilienceMetadataDTO, snapshots
│   │   ├── health_dto.py        ← HealthResponseDTO
│   │   ├── eda_dto.py           ← EdaSummaryResponseDTO, NumericalFeatureStatsDTO
│   │   ├── training_dto.py      ← TrainingRequestDTO, TrainingResponseDTO, leaderboard
│   │   └── prediction_dto.py    ← PredictionRequestDTO, PredictionResponseDTO, risk
│   │
│   ├── services/
│   │   ├── dataset_service.py   ← CSV I/O, schema validation, deduplication
│   │   ├── eda_service.py       ← Statistical profiling, skewness, domain insights
│   │   ├── preprocessing_service.py ← Pipeline builder (OHE, PowerTransformer, interactions)
│   │   ├── training_service.py  ← Cross-validation, ranking, artefact persistence
│   │   ├── prediction_service.py← Lazy-load inference, BMI categorisation, risk tiers
│   │   └── fallback_service.py  ← Actuarial Heuristic Fallback Engine
│   │
│   ├── controllers/
│   │   ├── health_controller.py ← GET /health
│   │   ├── eda_controller.py    ← GET /eda/summary
│   │   ├── training_controller.py ← POST /model/train, GET /model/metrics
│   │   └── prediction_controller.py ← POST /predict
│   │
│   ├── data/
│   │   └── insurance.csv        ← Kaggle dataset (1338 rows × 7 columns)
│   ├── artifacts/               ← Joblib artefacts + JSON evaluation report
│   └── main.py                  ← FastAPI factory + lifespan + middleware
│
├── tests/
│   └── test_api.py              ← 6-test pytest suite (TestClient + circuit forcing)
├── .env                         ← All 80+ configuration values
└── requirements.txt
```

### Request Flow

```
HTTP Request
    │
    ▼
SLF4J MDC Trace Middleware  ← injects X-Trace-Id, logs request/response
    │
    ▼
FastAPI Router  (/api/v1/...)
    │
    ▼
Controller (async route handler)
    │
    ▼
@resilience4j_pipeline wrapper (inner async helper)
    │
    ├─ Fallback (outermost)
    │   ├─ CircuitBreaker.allow_request()
    │   │   ├─ RateLimiter.acquire_permission()
    │   │   │   └─ TimeLimiter.execute()
    │   │   │       └─ Bulkhead.acquire()
    │   │   │           └─ Retry.execute()
    │   │   │               └─ Service Method (business logic)
    │   │   └─ record_success() / record_failure()
    │   └─ fallback_callable() on exception (if enabled)
    │
    ▼
PredictionResponseDTO / TrainingResponseDTO / EdaSummaryResponseDTO
    (with resilience_metadata auto-injected by _inject_metadata)
    │
    ▼
JSON Response + X-Trace-Id header
```

---

## Core ML Tasks

### 1. Explore & Preprocess Dataset

The `PreprocessingService` builds a **scikit-learn `Pipeline`** with the following steps:

| Step | Component | Purpose |
|------|-----------|---------|
| Feature engineering | `FunctionTransformer` | `bmi_smoker_interaction`, `is_obese`, `age_bmi_interaction` |
| Binary encoding | `OneHotEncoder(drop="if_binary")` | `smoker` (yes/no → 0/1) |
| Nominal encoding | `OneHotEncoder(handle_unknown="ignore")` | `sex`, `region` |
| Numerical transform | `PowerTransformer(method="yeo-johnson")` | Corrects right-skew on `bmi`, `age`, `children` |
| Scaling | `StandardScaler` | Z-score normalization |
| Target transform | `TransformedTargetRegressor` | `log(1+y)` → `exp(y)−1` on `charges` (right-skewed target) |

**Duplicate removal:** configurable via `drop_duplicates` (default: `True`).  
**Skewness threshold:** `|skewness| ≥ SKEWNESS_HIGH_THRESHOLD` (default: `0.75`).

### 2. Train & Compare Models

Three scikit-learn regressors are trained in parallel via `cross_validate`:

| Model | Key Hyperparameters (from .env) |
|-------|--------------------------------|
| `LinearRegression` | `fit_intercept=True` |
| `RandomForestRegressor` | `n_estimators=200, max_depth=8, min_samples_split=5` |
| `GradientBoostingRegressor` | `n_estimators=150, learning_rate=0.05, max_depth=4, subsample=0.85` |

### 3. K-Fold Cross-Validation

All models are evaluated with `KFold(n_splits=5, shuffle=True, random_state=42)`:

- **Metrics reported:** `MAE (mean ± std)`, `RMSE (mean ± std)`, `R² (mean ± std)`
- **Ranking criterion:** Highest `r2_mean` → lowest `rmse_mean` (tie-breaker)
- **Best pipeline** is fit on the full `(X, y)` and persisted via `joblib`
- **Evaluation report** saved as `app/artifacts/evaluation_report.json`

### 4. Predict Insurance Cost

The `POST /api/v1/predict` endpoint:

1. Validates request via `PredictionRequestDTO` (age 18–100, BMI 10–65, etc.)
2. Builds a 1-row `pd.DataFrame` with the input features
3. Runs `pipeline.predict(input_df)[0]` using the best trained pipeline
4. Classifies BMI: `Underweight < 18.5 ≤ Normal < 25.0 ≤ Overweight < 30.0 ≤ Obese`
5. Assigns risk tier:
   - `HIGH_RISK`: smoker
   - `MODERATE_RISK`: obese (BMI ≥ 30) OR senior (age ≥ 50)
   - `LOW_RISK`: none of the above
6. Returns `PredictionResponseDTO` with charge, formatted string, risk assessment, and Resilience4j telemetry

---

## Fault-Tolerance Matrix

| Pattern | Profile | Key Parameter | Behavior |
|---------|---------|---------------|----------|
| **Circuit Breaker** | predict | `failure_rate_threshold=50%`, `window=10`, `wait=30s` | Opens after ≥50% failures; auto-recovers via HALF_OPEN probe |
| **Circuit Breaker** | train | `failure_rate_threshold=50%`, `window=5`, `wait=60s` | Higher wait — training is expensive |
| **Circuit Breaker** | eda | `failure_rate_threshold=60%`, `window=10`, `wait=20s` | More lenient; EDA is stateless |
| **Rate Limiter** | predict | `60 req/60s` | Prevents prediction burst |
| **Rate Limiter** | train | `2 req/60s` | Training is CPU-intensive |
| **Rate Limiter** | eda | `30 req/60s` | Moderate EDA limit |
| **Bulkhead** | predict | `max_concurrent=20, wait=0.5s` | Caps concurrency to protect CPU |
| **Bulkhead** | train | `max_concurrent=1, wait=0s` | Only one training job at a time |
| **Bulkhead** | eda | `max_concurrent=10, wait=0.5s` | |
| **Time Limiter** | predict | `timeout=2.0s` | Fast SLA for real-time inference |
| **Time Limiter** | train | `timeout=30.0s` | Training can be slow |
| **Time Limiter** | eda | `timeout=5.0s` | |
| **Retry** | all | `max_attempts=3, wait=0.2s, backoff=2x` | Exponential backoff on transient failures |
| **Fallback** | predict | `ENABLE_FALLBACK_ON_PREDICT=True` | Actuarial Heuristic Engine returns estimated charge |
| **Fallback** | eda/metrics | `ENABLE_FALLBACK_ON_METRICS=True` | Degraded `TrainingResponseDTO` with empty leaderboard |

### Actuarial Heuristic Fallback Formula

When the ML pipeline is unavailable, the fallback computes:

```
estimated = BASE_CHARGE
          + age × AGE_MULTIPLIER
          + max(0, bmi - BMI_BASELINE) × EXCESS_BMI_MULTIPLIER
          + children × CHILDREN_MULTIPLIER
          + SMOKER_BASE_PENALTY           (if smoker == "yes")
          + SMOKER_OBESE_SYNERGY_PENALTY  (if smoker == "yes" AND bmi >= BMI_OBESE_THRESHOLD)
```

All coefficients configurable in `.env` Group 6.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Welcome page with links to docs and health |
| `GET` | `/docs` | Swagger UI (interactive) |
| `GET` | `/api/v1/health` | Full service + Resilience4j health check |
| `GET` | `/api/v1/eda/summary` | Dataset EDA with statistical profiling |
| `POST` | `/api/v1/model/train` | Trigger cross-validated training run |
| `GET` | `/api/v1/model/metrics` | Retrieve saved evaluation leaderboard |
| `POST` | `/api/v1/predict` | Real-time insurance cost prediction |

---

## Quickstart

### Prerequisites

- Python 3.11+
- The virtual environment at `pre_gdg/` (already created)

### 1. Activate Virtual Environment

```powershell
# Windows PowerShell
.\pre_gdg\Scripts\Activate.ps1

# Or Command Prompt
.\pre_gdg\Scripts\activate.bat
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Start the Server

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On startup, the service automatically:
1. Checks dataset availability at `app/data/insurance.csv`
2. Tries to load existing trained artefacts from `app/artifacts/`
3. If no artefacts exist, runs an initial training pass automatically

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

### 4. Run the Test Suite

```powershell
.\pre_gdg\Scripts\python.exe -m pytest -v
```

All 6 tests should pass. The test suite auto-trains the model on first run.

---

## Example Requests & Responses

### GET /api/v1/health

```bash
curl -s http://localhost:8000/api/v1/health | python -m json.tool
```

```json
{
  "status": "UP",
  "dataset_available": true,
  "dataset_path": "app/data/insurance.csv",
  "model_trained": true,
  "active_best_model": "GradientBoostingRegressor",
  "last_trained_at": "2026-10-24T07:21:45.123456+00:00",
  "resilience4j_registry": {
    "predict": {
      "circuit_breaker": { "name": "predict-cb", "state": "CLOSED", "failure_rate_percent": 0.0 },
      "rate_limiter": { "name": "predict-rl", "available_permissions": 60 },
      "bulkhead": { "name": "predict-bh", "available_concurrent_calls": 20 }
    },
    "train": { "..." : "..." },
    "eda": { "..." : "..." }
  }
}
```

### GET /api/v1/eda/summary

```bash
curl -s http://localhost:8000/api/v1/eda/summary | python -m json.tool
```

```json
{
  "total_rows": 1338,
  "total_columns": 7,
  "duplicate_rows": 1,
  "missing_values": { "age": 0, "sex": 0, "bmi": 0, "children": 0, "smoker": 0, "region": 0, "charges": 0 },
  "numerical_summary": {
    "charges": {
      "mean": 13270.4223,
      "std": 12110.0117,
      "min": 1121.8739,
      "max": 63770.428,
      "skewness": 1.5142,
      "is_highly_skewed": true
    }
  },
  "domain_insights": {
    "avg_charge_smoker": 32050.23,
    "avg_charge_non_smoker": 8434.27,
    "smoker_cost_multiplier": 3.8,
    "correlation_with_charges": { "age": 0.299, "bmi": 0.198, "children": 0.068 }
  },
  "resilience_metadata": {
    "circuit_breaker_state": "CLOSED",
    "fallback_triggered": false,
    "execution_time_ms": 12.45
  }
}
```

### POST /api/v1/model/train

```bash
curl -s -X POST http://localhost:8000/api/v1/model/train \
  -H "Content-Type: application/json" \
  -d '{"cv_folds": 5, "drop_duplicates": true, "handle_feature_skewness": true, "apply_log_transform_on_target": true, "include_bmi_smoker_interaction": true}' \
  | python -m json.tool
```

```json
{
  "status": "SUCCESS",
  "trained_at": "2026-10-24T07:21:45.123456+00:00",
  "cv_folds": 5,
  "best_model_selected": "GradientBoostingRegressor",
  "selection_criterion": "Highest mean R² across CV folds (lowest RMSE on tie)",
  "leaderboard": [
    {
      "rank": 1,
      "model_name": "GradientBoostingRegressor",
      "metrics": { "r2_mean": 0.8712, "r2_std": 0.0124, "rmse_mean": 4312.45, "rmse_std": 289.1 },
      "is_best": true
    },
    {
      "rank": 2,
      "model_name": "RandomForestRegressor",
      "metrics": { "r2_mean": 0.8601, "r2_std": 0.0198, "rmse_mean": 4521.12, "rmse_std": 341.2 },
      "is_best": false
    },
    {
      "rank": 3,
      "model_name": "LinearRegression",
      "metrics": { "r2_mean": 0.7489, "r2_std": 0.0215, "rmse_mean": 5912.33, "rmse_std": 412.7 },
      "is_best": false
    }
  ]
}
```

### POST /api/v1/predict

**High-risk smoker:**

```bash
curl -s -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 34, "sex": "male", "bmi": 31.4, "children": 2, "smoker": "yes", "region": "southeast"}' \
  | python -m json.tool
```

```json
{
  "predicted_charge": 28412.87,
  "currency": "USD",
  "formatted_charge": "$28,412.87",
  "model_used": "GradientBoostingRegressor",
  "input_summary": {
    "age": 34,
    "sex": "male",
    "bmi": 31.4,
    "bmi_category": "Obese",
    "children": 2,
    "smoker": "yes",
    "region": "southeast"
  },
  "risk_assessment": {
    "risk_tier": "HIGH_RISK",
    "key_cost_drivers": [
      "Smoking status",
      "Obesity (BMI ≥ 30.0)",
      "Number of dependents (2)"
    ]
  },
  "resilience_metadata": {
    "circuit_breaker_name": "predict-cb",
    "circuit_breaker_state": "CLOSED",
    "fallback_triggered": false,
    "retry_attempts_used": 0,
    "execution_time_ms": 8.72
  }
}
```

**Low-risk non-smoker:**

```bash
curl -s -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 22, "sex": "female", "bmi": 21.5, "children": 0, "smoker": "no", "region": "northwest"}' \
  | python -m json.tool
```

```json
{
  "predicted_charge": 2843.19,
  "currency": "USD",
  "formatted_charge": "$2,843.19",
  "model_used": "GradientBoostingRegressor",
  "risk_assessment": {
    "risk_tier": "LOW_RISK",
    "key_cost_drivers": ["No significant individual risk factors identified"]
  },
  "resilience_metadata": {
    "circuit_breaker_state": "CLOSED",
    "fallback_triggered": false
  }
}
```

---

## Configuration Reference

All settings live in `.env`. Key groups:

| Group | Variables | Description |
|-------|-----------|-------------|
| 1. Application | `APP_NAME`, `APP_VERSION`, `API_V1_PREFIX` | Service identity |
| 2. Paths | `DATASET_FILE_PATH`, `ARTIFACTS_DIR`, `*_FILENAME` | File locations |
| 3. Domain | `BMI_*_THRESHOLD`, `AGE_SENIOR_RISK_THRESHOLD`, `SKEWNESS_HIGH_THRESHOLD` | Medical thresholds |
| 4. Validation | `VALIDATION_AGE_MIN/MAX`, `VALIDATION_BMI_MIN/MAX` | DTO bounds |
| 5. ML | `DEFAULT_CV_FOLDS`, `RF_*`, `GB_*`, `LR_*` | Hyperparameters |
| 6. Fallback | `FALLBACK_BASE_CHARGE`, `FALLBACK_*_MULTIPLIER` | Actuarial formula weights |
| 7. Circuit Breaker | `CB_*_FAILURE_RATE_THRESHOLD`, `CB_*_SLIDING_WINDOW_SIZE` | CB tuning |
| 8. Rate Limiter | `RL_*_LIMIT_FOR_PERIOD`, `RL_*_REFRESH_PERIOD_SEC` | RL tuning |
| 9. Bulkhead | `BH_*_MAX_CONCURRENT_CALLS`, `BH_*_MAX_WAIT_DURATION_SEC` | BH tuning |
| 10. Time Limiter | `TL_*_TIMEOUT_SEC` | TL tuning |
| 11. Retry/Fallback | `RETRY_MAX_ATTEMPTS`, `ENABLE_FALLBACK_ON_PREDICT` | Retry + fallback toggles |
| 12. Logging | `LOG_LEVEL`, `LOG_FORMAT`, `LOG_DATE_FORMAT` | SLF4J MDC format |

---


