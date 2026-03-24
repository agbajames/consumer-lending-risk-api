from typing import Any, List, Optional, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

from .config import RANDOM_STATE


def split_feature_types(X) -> Tuple[List[str], List[str]]:
    """Split columns into numeric and categorical lists."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def _make_estimator(*, backend: str, y: Optional[Any] = None):
    """Create a base estimator for the given backend."""
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.0552,
            num_leaves=28,
            max_depth=9,
            subsample=0.744,
            colsample_bytree=0.811,
            reg_alpha=5.12e-07,
            reg_lambda=3.58e-04,
            min_child_samples=20,
            class_weight="balanced",  # handles imbalance
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif backend == "xgboost":
        from xgboost import XGBClassifier

        spw = 1.0
        if y is not None:
            # Standard scale_pos_weight = neg / pos
            pos = float((y == 1).sum())
            neg = float((y == 0).sum())
            spw = (neg + 1e-6) / (pos + 1e-6)

        return XGBClassifier(
            n_estimators=900,
            learning_rate=0.0204,
            max_depth=6,
            subsample=0.631,
            colsample_bytree=0.884,
            reg_alpha=7.72e-06,
            reg_lambda=5.390,
            min_child_weight=2,
            gamma=0.0513,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=spw,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")


def build_pipeline(
    numeric: List[str],
    categorical: List[str],
    *,
    backend: str = "lightgbm",
    y: Optional[Any] = None,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 5,
) -> Pipeline:
    """
    Build a full pipeline with preprocessing + calibrated classifier.

    backend ∈ {"lightgbm", "xgboost"}.
    y is used to compute scale_pos_weight for XGBoost.
    """
    # Preprocessing – impute only (trees do not need scaling)
    num_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])

    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ],
        remainder="drop",
    )

    base = _make_estimator(backend=backend, y=y)

    clf = CalibratedClassifierCV(
        estimator=base, cv=calibration_cv, method=calibration_method
    )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])
