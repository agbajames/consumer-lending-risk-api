import numpy as np
from fastapi.testclient import TestClient
from src import api


def _sample_payload():
    """Standard test payload."""
    return {
        "year": 2017,
        "loan_limit": "cf",
        "Gender": "Male",
        "approv_in_adv": "nopre",
        "loan_type": "type1",
        "loan_purpose": "p1",
        "Credit_Worthiness": "l1",
        "open_credit": "yes",
        "business_or_commercial": "no",
        "loan_amount": 150000.0,
        "rate_of_interest": 8.5,
        "Interest_rate_spread": 1.2,
        "Upfront_charges": 1200.0,
        "term": 180.0,
        "Neg_ammortization": "no",
        "interest_only": "no",
        "lump_sum_payment": "no",
        "property_value": 220000.0,
        "construction_type": "existing",
        "occupancy_type": "owner",
        "Secured_by": "home",
        "total_units": "1",
        "income": 6500.0,
        "credit_type": "CR1",
        "Credit_Score": 720.0,
        "co_applicant_credit_type": "Na",
        "age": "25-34",
        "submission_of_application": "to_inst",
        "LTV": 68.0,
        "Region": "south",
        "Security_Type": "type1",
        "dtir1": 28.0,
    }


def test_health_when_unloaded():
    """Test health endpoint when model isn't loaded."""
    client = TestClient(api.app)
    api._pipeline = None
    api._metadata = {}
    
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "error"


def test_score_ok_with_mocked_model(monkeypatch):
    """Test scoring with a mocked model."""
    client = TestClient(api.app)
    
    class Dummy:
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])
    
    api._pipeline = Dummy()
    api._metadata = {
        "features": list(_sample_payload().keys()),
        "trained_at": "2024-01-01T00:00:00Z",
        "model_type": "Calibrated(lightgbm)",
        "backend": "lightgbm",
        "threshold": 0.5,
    }
    
    r = client.post("/score", json=_sample_payload())
    assert r.status_code == 200
    
    js = r.json()
    assert 0 <= js["default_probability"] <= 1
    assert js["prediction"] in (0, 1)


def test_missing_feature_returns_400():
    """Test that missing features return clear error."""
    client = TestClient(api.app)
    
    class Dummy:
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])
    
    api._pipeline = Dummy()
    payload = _sample_payload()
    feats = list(payload.keys())
    api._metadata = {"features": feats}
    
    # Drop one required feature
    payload.pop(feats[0])
    
    r = client.post("/score", json=payload)
    assert r.status_code == 422