import logging

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
    LEAKY_COLS,
)

logger = logging.getLogger(__name__)

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
    - Drop leaky post-origination columns (missingness ≈ target proxy).
    - Drop rows with missing target.

    NOTE: Imputation is handled exclusively by the sklearn pipeline
    to prevent train/test leakage.
    """
    df = df.copy()

    # Rename columns as needed
    df = df.rename(columns=RENAME_MAP)

    # Drop ID columns if present
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Drop leaky columns whose missingness pattern is a target proxy
    leaky_present = [c for c in LEAKY_COLS if c in df.columns]
    if leaky_present:
        df = df.drop(columns=leaky_present)
        logger.info("Dropped %d leaky columns: %s", len(leaky_present), leaky_present)

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COLUMN])

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

    logger.info("Saved %s  (%d rows)", PROCESSED_TRAIN, len(train))
    logger.info("Saved %s  (%d rows)", PROCESSED_TEST, len(test))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = load_raw_data()
    df = basic_clean(df)
    make_train_test_split(df)
