import argparse
import json
import time
import joblib
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from .config import (
    PROCESSED_TRAIN,
    PROCESSED_TEST,
    PIPELINE_PATH,
    METADATA_PATH,
    TARGET_COLUMN,
    DEFAULT_THRESHOLD,
)
from .model import build_pipeline, split_feature_types


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["lightgbm", "xgboost"],
        default="lightgbm",
        help="Model backend to use.",
    )
    args = parser.parse_args()
    
    # Load data
    train = pd.read_csv(PROCESSED_TRAIN)
    test = pd.read_csv(PROCESSED_TEST)
    
    X_train = train.drop(columns=[TARGET_COLUMN])
    y_train = train[TARGET_COLUMN].astype(int)
    
    X_test = test.drop(columns=[TARGET_COLUMN])
    y_test = test[TARGET_COLUMN].astype(int)
    
    # Split features
    numeric, categorical = split_feature_types(X_train)
    
    # Build pipeline
    pipeline = build_pipeline(
        numeric,
        categorical,
        backend=args.backend,
        y=y_train,
        calibration_cv=5,
    )
    
    # Train
    print(f"Training {args.backend} model...")
    pipeline.fit(X_train, y_train)
    
    # Predict
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= DEFAULT_THRESHOLD).astype(int)
    
    # Evaluate
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "threshold": float(DEFAULT_THRESHOLD),
    }
    
    # Save model
    joblib.dump(pipeline, PIPELINE_PATH)
    
    # Save metadata
    metadata = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_type": f"Calibrated({args.backend})",
        "backend": args.backend,
        "features": X_train.columns.tolist(),
        "numeric": numeric,
        "categorical": categorical,
        **metrics,
    }
    
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Saved: {PIPELINE_PATH}")
    print(f"✅ Saved: {METADATA_PATH}")
    print(f"\n📊 Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()