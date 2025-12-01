import pandas as pd
from src.data_prep import basic_clean
from src.config import TARGET_COLUMN


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