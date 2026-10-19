"""Evaluation in the units clinicians care about.

AUC is fine for papers. A screening program cares about:
- recall (sensitivity) at fixed specificity: "if we can only confirm 10%
  of patients with HPLC, how many carriers do we catch?"
- calibration: is the score usable as a direct referral probability?
- data efficiency: a district clinic has 200 labeled cases, not 12k.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def recall_at_specificity(y_true, y_score, specificity: float = 0.90) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= (1 - specificity)
    if not mask.any():
        return float("nan")
    return float(tpr[mask].max())


def operating_table(results: list, specificities=(0.80, 0.90, 0.95)) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"model": r.name, "auc": roc_auc_score(r.y_true, r.y_score)}
        for spec in specificities:
            row[f"recall@spec{int(spec*100)}"] = recall_at_specificity(
                r.y_true, r.y_score, spec)
        row["brier"] = brier_score_loss(r.y_true, r.y_score)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def learning_curve(model_factory, X: np.ndarray, y: np.ndarray,
                   train_sizes=(100, 300, 1000, 5000),
                   n_repeats: int = 10, eval_size: int = 3000,
                   seed: int = 0) -> pd.DataFrame:
    """Mean AUC at various labeled-data budgets.

    This is where zero-shot foundation models should shine: a district
    clinic labels 200 cases, not 12,000.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n_train in train_sizes:
        aucs = []
        for rep in range(n_repeats):
            idx = rng.permutation(len(y))
            tr = idx[:n_train]
            te = idx[n_train:n_train + eval_size]
            m = model_factory()
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        rows.append({"n_train": n_train,
                     "auc_mean": float(np.mean(aucs)),
                     "auc_std": float(np.std(aucs))})
    return pd.DataFrame(rows)
