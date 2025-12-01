from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA = DATA_DIR / "raw" / "Loan_Default.csv"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_TRAIN = PROCESSED_DIR / "train.csv"
PROCESSED_TEST = PROCESSED_DIR / "test.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
PIPELINE_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

# Modelling
TARGET_COLUMN = "Status"
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5

# Ensure directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)