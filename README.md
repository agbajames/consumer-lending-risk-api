# Consumer Lending Risk API

![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://github.com/agbajames/consumer-lending-risk-api/workflows/ci/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)

Production-ready credit default prediction API demonstrating enterprise MLOps patterns for regulated financial services. Built with LightGBM/XGBoost, FastAPI, and comprehensive testing infrastructure.

## Overview

This project implements an end-to-end machine learning system for consumer lending risk assessment, featuring:

- **Dual ML backends** (LightGBM/XGBoost) with configurable training
- **Probability calibration** for well-calibrated risk scores suitable for loan pricing
- **Class imbalance handling** via `scale_pos_weight` and `class_weight="balanced"`
- **Production-grade API** with FastAPI, Pydantic validation, and health monitoring
- **Feature contract enforcement** via metadata tracking to prevent train-serve skew
- **Comprehensive testing** with pytest (6 tests covering preprocessing, models, and API)
- **Containerization** with Docker and health checks
- **CI/CD automation** via GitHub Actions

## Quick Start

### Prerequisites

- Python 3.11+
- Virtual environment tool (venv, conda, etc.)

### Setup

1. **Download the dataset** from [Kaggle: Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)
2. **Place** `Loan_Default.csv` in `data/raw/`
3. **Create virtual environment and install dependencies:**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Pipeline
```bash
# 1. Prepare data (clean, split into train/test)
python -m src.data_prep

# 2. Train model (choose backend: lightgbm or xgboost)
python -m src.train --backend lightgbm

# 3. Start API server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# 4. Test the API (in another terminal)
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d @sample_request.json

# 5. Run tests
pytest -v
```

### Using Make Commands
```bash
make prepare  # Install dependencies
make data     # Prepare datasets
make train    # Train model (default: lightgbm)
make serve    # Start API server
make test     # Run tests
```

## API Endpoints

### `GET /health`
Returns model status and performance metrics.

**Response:**
```json
{
  "status": "ok",
  "model_trained_at": "2024-12-01T13:17:16Z",
  "model_type": "Calibrated(lightgbm)",
  "backend": "lightgbm",
  "roc_auc": 1.0,
  "pr_auc": 1.0,
  "threshold": 0.5
}
```

### `POST /score`
Scores a loan application and returns default probability.

**Request body:** See `sample_request.json` for schema

**Response:**
```json
{
  "default_probability": 0.1524,
  "prediction": 0,
  "threshold": 0.5,
  "model_info": {
    "model_type": "Calibrated(lightgbm)",
    "backend": "lightgbm",
    "trained_at": "2024-12-01T13:17:16Z"
  }
}
```

## Project Structure
```
consumer-lending-risk-api/
├── data/
│   ├── raw/              # Original dataset (not committed)
│   └── processed/        # Train/test splits (generated)
├── artifacts/            # Trained models + metadata (generated)
├── src/                  # Source code
│   ├── __init__.py
│   ├── config.py         # Centralized configuration
│   ├── data_prep.py      # Data preprocessing pipeline
│   ├── model.py          # ML model construction
│   ├── train.py          # Training script
│   ├── api.py            # FastAPI application
│   └── schemas.py        # Pydantic schemas
├── tests/                # Automated tests
│   ├── test_api.py
│   ├── test_data_prep.py
│   └── test_pipeline.py
├── docs/
│   └── PLAYBOOK.md       # Detailed documentation
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI/CD
├── Dockerfile            # Container image definition
├── Makefile              # Convenient commands
├── requirements.txt      # Python dependencies
├── sample_request.json   # Example API request
└── README.md
```

## Key Features

### ML Engineering

- **Modular preprocessing pipelines** using scikit-learn's `ColumnTransformer`
- **Separate numeric/categorical handling** with appropriate imputation strategies
- **Feature contract validation** ensuring API requests match training schema
- **Model metadata tracking** for reproducibility and governance

### Production Patterns

- **Probability calibration** via `CalibratedClassifierCV` for reliable risk scores
- **Class imbalance mitigation** preventing majority-class prediction bias
- **Health monitoring endpoints** for operational observability
- **Comprehensive error handling** with clear validation messages
- **Docker containerization** with non-root user and health checks
- **CI/CD automation** running tests on every commit

### Testing

- **Unit tests** for data preprocessing logic
- **Integration tests** for model pipeline (both backends)
- **API tests** with mocked models for fast execution
- **96%+ code coverage** across critical paths

## Technology Stack

**Core ML:**
- Python 3.11
- scikit-learn 1.3.2
- Pandas 2.1.1, NumPy 1.26.4
- LightGBM 4.5.0
- XGBoost 2.1.1

**API & Deployment:**
- FastAPI 0.110.0
- Pydantic 2.7.4
- Uvicorn 0.23.2
- Docker

**Testing & CI/CD:**
- pytest 7.4.2
- GitHub Actions

## Model Performance

**Note:** This project uses a simplified Kaggle dataset optimized for learning MLOps patterns. The model achieves near-perfect metrics (ROC-AUC 1.0) due to high data separability. 

**Focus areas:**
- Production-grade infrastructure and deployment patterns
- Model governance and feature contract enforcement
- Comprehensive testing and CI/CD automation
- Patterns applicable to more complex, real-world datasets

For a more realistic credit risk challenge, consider datasets like:
- [Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
- [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)

## Docker Deployment
```bash
# Build image
docker build -t credit-risk-api .

# Run container
docker run -p 8000:8000 credit-risk-api

# Test health endpoint
curl http://localhost:8000/health
```

## Development
```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests with coverage
pytest --cov=src --cov-report=html

# Format code
black src/ tests/

# Type checking
mypy src/
```

## Documentation

- **Playbook:** See `docs/PLAYBOOK.md` for detailed operational guide
- **API Docs:** Visit `http://localhost:8000/docs` when server is running
- **Architecture:** Feature contracts, calibration, and deployment patterns detailed in code comments

## License

MIT License - see `LICENSE` file for details

## Author

**James Agba**  
AI/ML Engineer | Data Scientist  
[LinkedIn](https://linkedin.com/in/agbajames) | [Email](mailto:agbajames@gmail.com)

## Acknowledgments

- Dataset: [Kaggle Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)
- Built as a demonstration of production ML engineering patterns for financial services

---

**Interested in collaborating or have questions?** Open an issue or reach out directly!