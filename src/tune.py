"""
Hyperparameter tuning via Optuna TPE sweep.

Optimises PR-AUC using 5-fold stratified CV on the raw estimator
(pre-calibration), then writes the best params to artifacts/best_params.json.

Usage:
    python -m src.tune --backend lightgbm --trials 50
    python -m src.tune --backend xgboost  --trials 50

The resulting artifacts/best_params.json is consumed by model.py when present,
falling back to the hardcoded defaults if not.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*X does not have valid feature names.*",
    category=UserWarning,
    module="sklearn",
)

import argparse
import json
import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    ARTIFACTS_DIR,
    PROCESSED_TRAIN,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from .model import split_feature_types

logger = logging.getLogger(__name__)

# ── Search spaces ──────────────────────────────────────────────────────────────


def _lgbm_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }


def _xgb_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
    }


# ── Objective ──────────────────────────────────────────────────────────────────


def _make_objective(
    backend: str, X: pd.DataFrame, y: pd.Series, numeric: list, categorical: list
):
    """Return an Optuna objective function closed over the training data."""

    def objective(trial: optuna.Trial) -> float:
        params = _lgbm_space(trial) if backend == "lightgbm" else _xgb_space(trial)

        # Build estimator with trial params
        if backend == "lightgbm":
            from lightgbm import LGBMClassifier

            estimator = LGBMClassifier(
                **params,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            )
        else:
            from xgboost import XGBClassifier

            pos = float((y == 1).sum())
            neg = float((y == 0).sum())
            estimator = XGBClassifier(
                **params,
                scale_pos_weight=(neg + 1e-6) / (pos + 1e-6),
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="logloss",
                verbosity=0,
            )

        # Preprocessing (same as production pipeline)
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        pre = ColumnTransformer(
            [
                ("num", num_pipe, numeric),
                ("cat", cat_pipe, categorical),
            ],
            remainder="drop",
        )

        pipe = Pipeline([("pre", pre), ("clf", estimator)])

        # 5-fold stratified CV, optimising PR-AUC
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_val)[:, 1]
            scores.append(average_precision_score(y_val, proba))

        return float(np.mean(scores))

    return objective


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["lightgbm", "xgboost"], default="lightgbm"
    )
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    train = pd.read_csv(PROCESSED_TRAIN)
    X = train.drop(columns=[TARGET_COLUMN])
    y = train[TARGET_COLUMN].astype(int)
    numeric, categorical = split_feature_types(X)

    logger.info(
        "Starting Optuna TPE sweep: backend=%s, trials=%d", args.backend, args.trials
    )

    # Silence Optuna's own logs; use INFO only for trial completions
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        _make_objective(args.backend, X, y, numeric, categorical),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    logger.info("Best PR-AUC (CV): %.4f", best.value)
    logger.info("Best params: %s", best.params)

    # Persist for model.py to consume
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS_DIR / "best_params.json"

    # Load existing file so the other backend's results are preserved
    existing: dict = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)

    existing[args.backend] = {"pr_auc_cv": best.value, "params": best.params}

    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info("Saved best params to %s", output_path)
    logger.info(
        "Next step: update model.py with these values, or set USE_TUNED_PARAMS=true"
    )


if __name__ == "__main__":
    main()
