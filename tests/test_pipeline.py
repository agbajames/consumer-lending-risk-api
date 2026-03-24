import numpy as np
import pandas as pd
import pytest
from src.model import build_pipeline, split_feature_types


@pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
def test_pipeline_fits_and_predicts(backend):
    """Test that both backends can fit and predict."""
    # Small synthetic dataset
    df = pd.DataFrame(
        {
            "year": [2017, 2018, 2017, 2019, 2018, 2017],
            "loan_amount": [100000, 150000, 120000, 200000, 180000, 90000],
            "Gender": ["Male", "Female", "Male", "Female", "Female", "Male"],
            "Status": [0, 1, 0, 1, 0, 1],
        }
    )
    X = df.drop(columns=["Status"])
    y = df["Status"].astype(int)

    num, cat = split_feature_types(X)
    pipe = build_pipeline(num, cat, backend=backend, y=y, calibration_cv=2)

    # Should fit without error
    pipe.fit(X, y)

    # Should predict probabilities
    proba = pipe.predict_proba(X)[:, 1]

    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))
