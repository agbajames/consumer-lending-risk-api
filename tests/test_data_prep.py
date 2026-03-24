import pandas as pd
import numpy as np
from src.data_prep import basic_clean
from src.config import TARGET_COLUMN, LEAKY_COLS


def test_rename_map_applied_and_id_dropped():
    """Test that column renaming works and ID columns are dropped."""
    df = pd.DataFrame(
        {
            "co-applicant_credit_type": ["Na", "CR1"],
            TARGET_COLUMN: [0, 1],
            "ID": [1, 2],
        }
    )
    out = basic_clean(df)

    # Check rename happened
    assert "co_applicant_credit_type" in out.columns
    assert "co-applicant_credit_type" not in out.columns

    # Check ID dropped
    assert "ID" not in out.columns

    # Check target preserved
    assert out[TARGET_COLUMN].isna().sum() == 0


def test_missing_target_rows_dropped():
    """Rows with NaN target are removed; other NaNs are preserved for pipeline."""
    df = pd.DataFrame(
        {
            "loan_amount": [100_000, np.nan, 120_000, 90_000],
            TARGET_COLUMN: [0, 1, 1, np.nan],  # row 3 has no label — should be dropped
        }
    )
    out = basic_clean(df)
    # Row with NaN target must be gone
    assert len(out) == 3
    assert out[TARGET_COLUMN].isna().sum() == 0
    # NaN in a feature column must survive (imputation is the pipeline's job)
    assert out["loan_amount"].isna().sum() == 1


def test_leaky_columns_dropped():
    """Leaky post-origination columns are removed during cleaning."""
    df = pd.DataFrame(
        {
            "loan_amount": [100_000, 200_000],
            "rate_of_interest": [3.5, np.nan],
            "Interest_rate_spread": [0.5, np.nan],
            "Upfront_charges": [1200, np.nan],
            "property_value": [250_000, np.nan],
            "LTV": [40.0, np.nan],
            TARGET_COLUMN: [0, 1],
        }
    )
    out = basic_clean(df)

    for col in LEAKY_COLS:
        assert col not in out.columns, f"{col} should have been dropped"

    # Non-leaky columns survive
    assert "loan_amount" in out.columns
