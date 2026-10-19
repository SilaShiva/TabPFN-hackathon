"""Load and prepare the HPLC-based thalassemia screening dataset.

Source: Kaggle ``abhraghoshcmc/hplc-based-thalassemia-screening-data``
(13,031 patients screened at a thalassemia center in West Bengal, India).

The screening task we target is the one that matters at the point of care:
**beta thalassaemia carrier vs normal, from CBC parameters alone** (plus
age/sex). HPLC quantification of HbA2 is the confirmatory test -- expensive
and centralized. A CBC is $2 at any clinic. If a CBC-only model could
reliably triage who needs confirmatory HPLC, screening programs could
scale to the ~3.5% of Indians who carry a hemoglobin variant.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CBC_FEATURES = ["RBC", "HB", "MCV", "MCH", "MCHC", "RDWcv"]
DEMOGRAPHICS = ["age", "sex"]
FEATURES = CBC_FEATURES + DEMOGRAPHICS

CARRIER_LABEL = "Beta Thalassaemia carrier"


def load_dataset(zip_path: str | Path) -> pd.DataFrame:
    """Read the Kaggle zip and return a tidy frame with a binary target.

    Output columns: CBC_FEATURES + ['age', 'sex', 'diagnosis', 'y']
    where y == 1 for beta thalassaemia carriers.
    """
    z = zipfile.ZipFile(zip_path)
    name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
    raw = pd.read_csv(io.BytesIO(z.read(name)))

    df = pd.DataFrame()
    df["diagnosis"] = raw["Diagnosis"].astype(str).str.strip()

    # keep only the two classes every CBC-screening program must separate
    df = df[df["diagnosis"].isin(["Normal", CARRIER_LABEL])].copy()
    df["y"] = (df["diagnosis"] == CARRIER_LABEL).astype(int)

    for col in CBC_FEATURES:
        df[col] = pd.to_numeric(raw[col], errors="coerce")

    # "14 Yrs 11 month" -> 14.9; "18 Yrs 0 month" -> 18.0
    age_str = raw["Age"].astype(str)
    yrs = pd.to_numeric(age_str.str.extract(r"(\d+)\s*Yrs")[0], errors="coerce")
    months = pd.to_numeric(age_str.str.extract(r"(\d+)\s*month")[0], errors="coerce").fillna(0)
    df["age"] = yrs + months / 12.0

    df["sex"] = (raw["Gender"] == "Female").astype(int)

    df = df.dropna(subset=FEATURES)
    df = df[df["age"] >= 1]  # drop absurd parsed ages
    return df.reset_index(drop=True)


def published_indices(df: pd.DataFrame) -> pd.DataFrame:
    """The six CBC discrimination indices from the hematology literature.

    These are what clinics actually use today: fixed formulas, one per
    paper, each claiming to beat the others. Lower => suggests trait
    (microcytosis with high RBC count) for all of them.
    """
    out = pd.DataFrame(index=df.index)
    out["mentzer"] = df["MCV"] / df["RBC"]
    out["shine_lal"] = df["MCV"] ** 2 * df["RBC"] / 100
    out["england_fraser"] = df["MCV"] - df["RBC"] - 5 * df["HB"]
    out["green_king"] = df["MCV"] ** 2 * df["RDWcv"] / (100 * df["HB"])
    out["rdwi"] = df["MCV"] * df["RDWcv"] / df["RBC"]
    out["ehsani"] = df["MCV"] - 10 * df["RBC"]
    return out
