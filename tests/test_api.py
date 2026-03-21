import numpy as np
import pytest
from fastapi.testclient import TestClient
from src import api


def _sample_payload():
    """Standard test payload (matches schema after leaky column removal)."""
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
        "term": 180.0,
        "Neg_ammortization": "no",
        "interest_only": "no",
        "lump_sum_payment": "no",
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
        "Region": "south",
        "Security_Type": "type1",
        "dtir1": 28.0,
    }


class _DummyModel:
    """Minimal mock that mimics pipeline.predict_proba."""

    def __init__(self, p1: float = 0.7):
        self._p1 = p1

    def predict_proba(self, X):
        return np.array([[1 - self._p1, self._p1]])


@pytest.fixture()
def client(monkeypatch):
    """TestClient with model state reset via monkeypatch."""
    monkeypatch.setattr(api, "_pipeline", None)
    monkeypatch.setattr(api, "_metadata", {})
    return TestClient(api.app)


# ── Health endpoint ───────────────────────────────────────────


def test_health_when_unloaded(client, monkeypatch):
    """Health returns status=error when no model is loaded."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "error"


# ── Score endpoint ────────────────────────────────────────────


def test_score_ok_with_mocked_model(client, monkeypatch):
    """Scoring returns valid probability and prediction."""
    monkeypatch.setattr(api, "_pipeline", _DummyModel(0.7))
    monkeypatch.setattr(
        api,
        "_metadata",
        {
            "features": list(_sample_payload().keys()),
            "trained_at": "2024-01-01T00:00:00Z",
            "model_type": "Calibrated(lightgbm)",
            "backend": "lightgbm",
            "threshold": 0.5,
        },
    )

    r = client.post("/score", json=_sample_payload())
    assert r.status_code == 200

    js = r.json()
    assert 0 <= js["default_probability"] <= 1
    assert js["prediction"] in (0, 1)


def test_missing_schema_field_returns_422(client, monkeypatch):
    """Pydantic rejects a payload with a missing required field (422)."""
    monkeypatch.setattr(api, "_pipeline", _DummyModel())
    monkeypatch.setattr(
        api,
        "_metadata",
        {"features": list(_sample_payload().keys())},
    )

    payload = _sample_payload()
    payload.pop("loan_amount")  # required field

    r = client.post("/score", json=payload)
    assert r.status_code == 422


def test_feature_contract_drift_returns_400(client, monkeypatch):
    """
    Feature-contract check catches metadata drift:
    if training metadata expects a column the schema doesn't provide,
    the endpoint returns 400.
    """
    features_with_extra = list(_sample_payload().keys()) + ["new_training_col"]

    monkeypatch.setattr(api, "_pipeline", _DummyModel())
    monkeypatch.setattr(
        api,
        "_metadata",
        {"features": features_with_extra},
    )

    r = client.post("/score", json=_sample_payload())
    assert r.status_code == 400
    assert "missing features" in r.json()["detail"]


def test_score_returns_503_when_model_unloaded(client, monkeypatch):
    """Score returns 503 when pipeline is None."""
    r = client.post("/score", json=_sample_payload())
    assert r.status_code == 503
