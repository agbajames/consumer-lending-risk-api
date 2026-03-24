# Playbook

## Data

- `data/raw/Loan_Default.csv` (Kaggle dataset)
- Target: `Status` (1 = default, 0 = non-default)
- Any ID column (e.g. `ID`) is dropped
- Canonical rename: `co-applicant_credit_type` → `co_applicant_credit_type` to keep Python identifiers clean

## Commands
```bash
# Prepare data
python -m src.data_prep

# Train – choose backend (default: lightgbm)
python -m src.train --backend lightgbm
python -m src.train --backend xgboost

# Serve API
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## API

**GET /health** → model status + basic metrics

**POST /score** → default probability and binary prediction

The API:
- Uses a strict Pydantic schema for input
- Looks up the expected feature list from metadata.json
- Validates that all required features are present
- Returns clear error messages for invalid requests

## Testing
```bash
pytest -v
```

## Docker
```bash
# Build
docker build -t credit-risk-api .

# Run
docker run -p 8000:8000 credit-risk-api

# Test
curl http://localhost:8000/health
```