"""Offline tests: dataset prep, index polarity, evaluation math.

No network. python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thalscreen.dataset import (  # noqa: E402
    CBC_FEATURES,
    FEATURES,
    load_dataset,
    published_indices,
)
from thalscreen.evaluate import (  # noqa: E402
    learning_curve,
    operating_table,
    recall_at_specificity,
)
from thalscreen.baselines import OOFResult, index_auc  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "hplc_thal.zip"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_dataset(DATA)


# -- dataset ------------------------------------------------------------------

def test_load_shape_and_labels(df):
    assert len(df) > 10_000
    assert set(df["y"].unique()) == {0, 1}
    assert df["y"].mean() < 0.10  # imbalanced like real screening
    assert not df[FEATURES].isna().any().any()


def test_only_normal_and_carrier(df):
    assert set(df["diagnosis"].unique()) == {"Normal", "Beta Thalassaemia carrier"}


def test_age_parsed_fractional(df):
    assert df["age"].between(1, 100).all()


def test_sex_binary(df):
    assert set(df["sex"].unique()) == {0, 1}


def test_cbc_plausible_ranges(df):
    # the raw register contains rare data-entry outliers (HB up to 41,
    # MCV down to 8); the model must handle them as-is, so only assert the
    # columns parsed as numeric and cover their physically possible span
    assert df["HB"].between(1, 50).all()
    assert df["MCV"].between(5, 130).all()


# -- published indices ----------------------------------------------------------

def test_six_indices_present(df):
    idx = published_indices(df)
    assert list(idx.columns) == [
        "mentzer", "shine_lal", "england_fraser", "green_king", "rdwi", "ehsani"]
    assert idx.notna().all().all()


def test_index_auc_ordering_matches_literature(df):
    aucs = index_auc(df, df["y"])
    # Ehsani/Mentzer are consistently the strongest in published comparisons
    assert aucs.index[0] in ("ehsani", "mentzer")
    # green_king is consistently among the weakest on adult Indian cohorts
    assert aucs.idxmin() in ("green_king", "england_fraser", "rdwi")


# -- evaluation ------------------------------------------------------------------

def test_perfect_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.9])
    assert recall_at_specificity(y, p, 0.95) == 1.0
    r = OOFResult("perfect", y, p)
    assert operating_table([r])["auc"].iloc[0] == 1.0


def test_random_scores_near_half():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = rng.random(2000)
    r = OOFResult("random", y, p)
    assert 0.45 < r.auc < 0.55


def test_learning_curve_shapes():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] > 0).astype(int)
    from sklearn.linear_model import LogisticRegression
    curve = learning_curve(LogisticRegression, X, y,
                           train_sizes=(50, 100), n_repeats=2, eval_size=100)
    assert list(curve.columns) == ["n_train", "auc_mean", "auc_std"]
    assert len(curve) == 2
