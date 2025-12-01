from typing import List, Tuple, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV


def split_feature_types(X) -> Tuple[List[str], List[str]]:
    """Split columns into numeric and categorical lists."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def _make_estimator(*, backend: str, y: Optional[object]):
    """Create a base estimator for the given backend."""
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier
        
        return LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",  # handles imbalance
            random_state=42,
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
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=42,
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
    y: Optional[object] = None,
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
        sparse_threshold=0.0,
    )
    
    base = _make_estimator(backend=backend, y=y)
    
    clf = CalibratedClassifierCV(
        base_estimator=base, cv=calibration_cv, method=calibration_method
    )
    
    return Pipeline(steps=[("pre", pre), ("clf", clf)])