import pandas as pd
from sklearn.model_selection import train_test_split
from .config import (
    RAW_DATA,
    PROCESSED_TRAIN,
    PROCESSED_TEST,
    PROCESSED_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)

# Columns to drop (IDs etc.)
DROP_COLS = ["ID"]

# Canonical renames for train ↔ serve alignment
RENAME_MAP = {
    "co-applicant_credit_type": "co_applicant_credit_type",
}


def load_raw_data() -> pd.DataFrame:
    """Load the raw Kaggle Loan_Default dataset."""
    return pd.read_csv(RAW_DATA)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Rename columns for canonical schema.
    - Drop known ID columns.
    - Drop rows with missing target.
    - Simple median/mode imputation.
    """
    df = df.copy()
    
    # Rename columns as needed
    df = df.rename(columns=RENAME_MAP)
    
    # Drop ID columns if present
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    
    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COLUMN])
    
    # Numeric vs categorical
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    cats = [c for c in df.columns if c not in numeric]
    
    # Median impute numeric
    for c in numeric:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    
    # Mode impute categorical
    for c in cats:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    
    return df


def make_train_test_split(df: pd.DataFrame) -> None:
    """Create stratified train/test splits and save to disk."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    
    train = X_train.copy()
    train[TARGET_COLUMN] = y_train
    
    test = X_test.copy()
    test[TARGET_COLUMN] = y_test
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_TRAIN, index=False)
    test.to_csv(PROCESSED_TEST, index=False)
    
    print(f"Saved {PROCESSED_TRAIN}")
    print(f"Saved {PROCESSED_TEST}")


if __name__ == "__main__":
    df = load_raw_data()
    df = basic_clean(df)
    make_train_test_split(df)