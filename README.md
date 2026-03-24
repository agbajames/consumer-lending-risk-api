# Consumer Lending Risk API

![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://github.com/agbajames/consumer-lending-risk-api/workflows/ci/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)

Production-style credit default prediction API demonstrating robust ML engineering patterns for credit risk modelling. Built with LightGBM/XGBoost, FastAPI, and comprehensive automated testing.

> 📓 **[View rendered notebook](https://htmlpreview.github.io/?https://github.com/agbajames/consumer-lending-risk-api/blob/main/pipeline.html)** — EDA, leakage analysis, model evaluation, and threshold review.

## Overview

This project implements an end-to-end machine learning system for consumer lending risk assessment using a public loan application dataset. It focuses on a core real-world modelling problem in credit risk: producing reliable default probabilities while detecting and eliminating target leakage that can otherwise create misleadingly perfect performance.

The repository includes configurable model training, calibrated probability estimation, leakage-aware preprocessing, a validated FastAPI scoring layer, automated tests, and containerised deployment support.

## Why this project matters

- **Detects and removes severe target leakage** caused by post-origination fields whose missingness pattern acts as a near-perfect proxy for default
- **Trains calibrated LightGBM and XGBoost models** to produce more reliable default probabilities for downstream risk-based decisioning
- **Serves predictions through a validated API** with feature contract enforcement, health checks, Docker support, and continuous integration

## Core Features

- **Dual ML backends** – LightGBM and XGBoost with configurable training
- **Probability calibration** – calibrated default probabilities for more reliable downstream decisioning
- **Leakage detection and removal** – automatic exclusion of post-origination columns that create unrealistic performance
- **Class imbalance handling** – `class_weight="balanced"` and `scale_pos_weight`
- **Production-oriented API** – FastAPI service with Pydantic validation, domain-aware constraints, and health monitoring
- **Feature contract enforcement** – metadata-backed schema validation to prevent train – serve skew
- **Automated testing** – unit, integration, and API tests using pytest
- **Containerisation** – Docker image with health checks and non-root user
- **Continuous integration** – GitHub Actions workflow running the test suite on every commit

## Model Performance

Trained on the [Kaggle Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset) containing 148,670 applications with an approximate default rate of 24.6%.

| Backend | ROC-AUC | PR-AUC | Brier Score |
|---------|---------|--------|-------------|
| LightGBM | 0.8828 | 0.8229 | 0.0913 |
| XGBoost | 0.8815 | 0.8226 | 0.0913 |

Metrics are reported after leakage removal and probability calibration on a held-out test split.

## Leakage Analysis

A key challenge in this dataset is the presence of five post-origination columns:

- `rate_of_interest`
- `Interest_rate_spread`
- `Upfront_charges`
- `property_value`
- `LTV`

Their missingness patterns act as near-perfect target proxies. For example, `Interest_rate_spread` is missing for 100% of defaults and 0% of non-defaults. If retained, these fields produce an artificial ROC-AUC of 1.0.

The pipeline detects and removes these columns before modelling, producing the more realistic performance shown above. See `pipeline.ipynb` Section 5 for the full leakage analysis.

## Quick Start

### Prerequisites

- Python 3.11+
- Virtual environment tool such as `venv` or `conda`

### Setup

1. **Download the dataset** from [Kaggle: Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)
2. **Place** `Loan_Default.csv` in `data/raw/`
3. **Create a virtual environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# 1. Prepare data
python -m src.data_prep

# 2. Train model
python -m src.train --backend lightgbm

# 3. Start API server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# 4. Test the API
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d @sample_request.json

# 5. Run tests
pytest -v
```

### Make Commands

```bash
make prepare  # Install dependencies
make data     # Prepare datasets
make train    # Train model (default: lightgbm)
make serve    # Start API server
make test     # Run tests
```

> **Note:** `artifacts/` is gitignored, so a fresh clone contains no trained model. Run `make data && make train` before `make serve` — otherwise the API will start but return a 503 on `/score`.

## API Endpoints

### `GET /health`

Returns service status and model metadata.

**Example response:**

```json
{
  "status": "ok",
  "model_trained_at": "2026-03-21T19:18:24Z",
  "model_type": "Calibrated(lightgbm)",
  "backend": "lightgbm",
  "roc_auc": 0.8828,
  "pr_auc": 0.8229,
  "threshold": 0.5
}
```

### `POST /score`

Scores a loan application and returns default probability.

**Example request:**

```json
{
  "loan_amount": 165000,
  "income": 72000,
  "Credit_Score": 710
}
```

See `sample_request.json` for the full request schema.

**Example response:**

```json
{
  "default_probability": 0.266859,
  "prediction": 0,
  "threshold": 0.5,
  "model_info": {
    "model_type": "Calibrated(lightgbm)",
    "backend": "lightgbm",
    "trained_at": "2026-03-21T19:18:24Z"
  }
}
```

## Project Structure

```text
consumer-lending-risk-api/
├── data/
│   ├── raw/              # Original dataset (not committed)
│   └── processed/        # Train/test splits (generated)
├── artifacts/            # Trained models + metadata (generated)
├── src/
│   ├── __init__.py
│   ├── config.py         # Centralised configuration and leaky column list
│   ├── data_prep.py      # Data cleaning and leakage removal
│   ├── model.py          # ML model construction
│   ├── train.py          # Training script
│   ├── api.py            # FastAPI application
│   └── schemas.py        # Pydantic schemas with domain constraints
├── tests/
│   ├── test_api.py       # API tests
│   ├── test_data_prep.py # Data preparation tests
│   └── test_pipeline.py  # End-to-end training pipeline tests
├── docs/
│   └── PLAYBOOK.md       # Operational guide
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI workflow
├── pipeline.ipynb        # EDA, leakage analysis, training, and evaluation notebook
├── Dockerfile
├── Makefile
├── requirements.txt
├── sample_request.json
└── README.md
```

## Key Design Decisions

### Data Integrity

- **No pre-split imputation** – imputation is handled inside the scikit-learn pipeline using `SimpleImputer`, so statistics are learned from training data only
- **Leakage-aware cleaning** – post-origination columns with target-proxy missingness patterns are removed before modelling
- **Feature contract validation** – inference requests are checked against saved training metadata to catch schema drift

### ML Engineering

- **Modular preprocessing** – `ColumnTransformer` pipelines with separate numeric and categorical treatment
- **Probability calibration** – `CalibratedClassifierCV` using Platt scaling to improve probability reliability
- **Class imbalance mitigation** – backend-specific handling through `class_weight="balanced"` and `scale_pos_weight`
- **Model metadata persistence** – trained models are stored alongside metrics, feature lists, and training timestamps for reproducibility and governance

### API and Deployment

- **Domain-aware validation** – Pydantic schemas enforce field-level constraints such as `loan_amount >= 0` and `Credit_Score` between 0 and 900
- **Health monitoring** – `/health` exposes service and model metadata
- **Structured logging** – no ad hoc `print()` debugging in production paths
- **Container hardening basics** – non-root user, health checks, and OpenMP support in Docker image

## Testing

The test suite covers unit, integration, and API behaviour:

- **Unit tests** – column renaming, ID removal, NaN preservation, and leaky column removal
- **Integration tests** – full pipeline fit and predict for both backends
- **API tests** – health endpoint, scoring, validation failures, feature contract drift, and model-not-loaded guards

Run the suite with:

```bash
pytest -v
```

For coverage:

```bash
pytest --cov=src --cov-report=html -v
```

## Docker Deployment

```bash
# Build image
docker build -t credit-risk-api .

# Run container
docker run -p 8000:8000 credit-risk-api

# Check health
curl http://localhost:8000/health
```

## Development

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run tests with coverage
pytest --cov=src --cov-report=html -v

# Format code
black src/ tests/

# Type checking
mypy src/
```

## Limitations

- This repository is a **technical demonstration** built on a public dataset and is not intended for direct production lending decisions without further governance and validation.
- Real-world credit deployment would require additional work on fairness testing, regulatory compliance, reject inference, out-of-time validation, monitoring, and model risk controls.
- The default classification threshold of `0.5` is illustrative and should be tuned to business costs, approval strategy, and risk appetite.
- Performance may vary materially under population drift or on operational data that differs from the benchmark dataset.

## Technology Stack

**Core ML**  
Python 3.11, scikit-learn 1.3.2, LightGBM 4.5.0, XGBoost 2.1.1, Pandas 2.1.1, NumPy 1.26.4

**API and Deployment**  
FastAPI 0.110.0, Pydantic 2.7.4, Uvicorn 0.23.2, Docker

**Testing and CI**  
pytest 7.4.2, GitHub Actions

## Documentation

- **Notebook:** `pipeline.ipynb` – EDA, leakage analysis, training, evaluation, and threshold review
- **Playbook:** `docs/PLAYBOOK.md` – operational guide
- **API docs:** available at `http://localhost:8000/docs` when the server is running

## License

MIT License – see `LICENSE` for details.

## Author

**James Agba**  
AI/ML Engineer | Data Scientist  
[LinkedIn](https://linkedin.com/in/agbajames) | [Email](mailto:agbajames@gmail.com)

## Acknowledgements

- Dataset: [Kaggle Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)
- Built as a demonstration of production-oriented ML engineering patterns for credit risk modelling
