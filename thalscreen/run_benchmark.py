"""Run the full screening benchmark end-to-end.

Produces results/benchmark_summary.md with every table and number.

    export TABPFN_TOKEN=tabpfn_sk_...
    python run_benchmark.py            # cloud inference (default)
    python run_benchmark.py --mode local
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from thalscreen import (
    FEATURES,
    MODELS,
    index_auc,
    learning_curve,
    load_dataset,
    operating_table,
    oof_predict,
)
from thalscreen.dataset import published_indices

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "hplc_thal.zip"
RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["client", "local"], default="client")
    ap.add_argument("--skip-curve", action="store_true",
                    help="skip the (slow) learning-curve section")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    df = load_dataset(DATA)
    y = df["y"].to_numpy()
    X = df[FEATURES].to_numpy()
    n_carriers = int(y.sum())

    lines = []
    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say(f"# CBC-only beta-thalassaemia carrier screening benchmark\n")
    say(f"Data: {len(df):,} patients "
        f"({n_carriers} carriers, {y.mean():.1%} prevalence), "
        f"West Bengal HPLC screening program (Kaggle: "
        f"abhraghoshcmc/hplc-based-thalassemia-screening-data)")
    say(f"Features: {', '.join(FEATURES)} (all available from a $2 CBC)\n")

    # -- 1. published indices -------------------------------------------------
    say("## 1. The six published CBC discrimination indices (clinical standard)\n")
    iauc = index_auc(df, df["y"])
    say("| Index | AUC |")
    say("|---|---|")
    for name, auc in iauc.items():
        say(f"| {name} | {auc:.4f} |")
    say("")

    # -- 2. ML baselines + TabPFN --------------------------------------------
    say(f"## 2. Models (5-fold stratified OOF, seed 42)\n")
    results = []
    for name in MODELS:
        results.append(oof_predict(name, X, y))

    from thalscreen.tabpfn_model import make_tabpfn
    tabpfn_oof = oof_predict(
        "TabPFN-3.5", X, y, model=make_tabpfn(args.mode))
    results.append(tabpfn_oof)

    table = operating_table(results)
    say(table.to_markdown(index=False, floatfmt=".4f"))
    say("")
    say("`recall@specNN` = sensitivity when the model flags at most NN% of "
        "patients for confirmatory HPLC. `brier` = calibration error "
        "(lower is better; the score is used as a referral probability).")

    # per-index table merged for context
    say("\n## 3. Best published index vs best model (head to head)\n")
    best_idx_name = iauc.index[0]
    best_idx_auc = iauc.iloc[0]
    tab_auc = table.loc[table["model"] == "TabPFN-3.5", "auc"].iloc[0]
    gb_auc = table.loc[table["model"] == "HistGradientBoosting", "auc"].iloc[0]
    say(f"- Best published index: **{best_idx_name}** at AUC {best_idx_auc:.4f}")
    say(f"- Best GBDT: HistGradientBoosting at AUC {gb_auc:.4f}")
    say(f"- **TabPFN-3.5: AUC {tab_auc:.4f}** "
        f"(+{tab_auc-best_idx_auc:.4f} over the clinical standard, "
        f"+{tab_auc-gb_auc:.4f} over the tuned GBDT)")

    # -- 4. learning curve -----------------------------------------------------
    if not args.skip_curve:
        say("\n## 4. Data efficiency (mean AUC over 10 random splits)\n")
        say("A district clinic labels a few hundred cases, not 12k.\n")
        curve_rows = []
        for name in ("HistGradientBoosting", "TabPFN-3.5"):
            if name == "TabPFN-3.5":
                factory = lambda: make_tabpfn(args.mode)
            else:
                factory = MODELS[name]
            curve = learning_curve(factory, X, y)
            curve["model"] = name
            curve_rows.append(curve)
        curve = pd.concat(curve_rows)
        piv = curve.pivot(index="n_train", columns="model", values="auc_mean")
        piv["delta"] = piv["TabPFN-3.5"] - piv["HistGradientBoosting"]
        say(piv.to_markdown(floatfmt=".4f"))

    out = RESULTS / "benchmark_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
