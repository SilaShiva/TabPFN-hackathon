"""Baseline models for the screening benchmark.

- the six published formula indices (what clinics use today)
- logistic regression (balanced)
- HistGradientBoosting (sklearn's GBDT, the de-facto tabular winner)

All evaluated identically: 5-fold stratified out-of-fold predictions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def index_auc(df: pd.DataFrame, y: pd.Series) -> pd.Series:
    """AUC of each published index (polarity-agnostic: higher AUC wins)."""
    from thalscreen.dataset import published_indices

    idx = published_indices(df)
    return pd.Series(
        {name: max(roc_auc_score(y, s), roc_auc_score(y, -s))
         for name, s in idx.items()}
    ).sort_values(ascending=False)


@dataclass
class OOFResult:
    name: str
    y_true: np.ndarray
    y_score: np.ndarray

    @property
    def auc(self) -> float:
        return roc_auc_score(self.y_true, self.y_score)


MODELS = {
    "LogisticRegression": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    ),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(random_state=42),
}


def oof_predict(name: str, X: np.ndarray, y: np.ndarray,
                model=None, n_splits: int = 5, seed: int = 42) -> OOFResult:
    """Out-of-fold predictions with fixed stratified 5-fold CV."""
    clf = model if model is not None else MODELS[name]()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return OOFResult(name=name, y_true=y, y_score=p)
