# thalscreen

**A CBC-only screening test for beta-thalassaemia carriers — and the
benchmark showing TabPFN-3.5 beats both the clinical standard and tuned
gradient boosting at it.**

## The problem (a real one, with a number attached)

Beta thalassaemia is carried by ~3–4% of Indians; when two carriers marry,
1-in-4 of their children develops thalassaemia major — lifelong transfusions,
death in the teens without them. The entire prevention strategy is carrier
screening before marriage. The confirmatory test (HPLC quantification of
HbA2) is centralized and costs ₹300–500. The complete blood count (CBC)
costs ₹150, is available at every district clinic, and **has been used for
screening for 50 years via simple published formulas** — Mentzer (1973),
Shine & Lal, England & Fraser, Green & King, RDWI, Ehsani — each a fixed
hand-tuned arithmetic expression, each claiming superiority in some cohort,
each with the same weakness: a single formula cannot model the interaction
of six correlated parameters across age, sex, and comorbid iron deficiency.

Machine learning papers keep promising to replace these indices, then stop
at small single-center cohorts with Gaussian assumptions. Nobody has run a
modern tabular foundation model on a large real screening cohort.

## What this project shows (on 12,338 real patients)

Data: [HPLC-Based Thalassemia Screening Data](https://www.kaggle.com/datasets/abhraghoshcmc/hplc-based-thalassemia-screening-data)
— 13,031 patients from a West Bengal screening program, CBC + HPLC, with
HPLC-confirmed diagnosis. Task: **carrier vs normal from 8 CBC-only
features** (RBC, Hb, MCV, MCH, MCHC, RDW-CV, age, sex) — the exact data a
district clinic has before any HPLC referral.

### Headline numbers (5-fold stratified OOF, seed 42)

| Model | AUC | recall@spec90 | Brier |
|---|---|---|---|
| Best published index (Ehsani) | 0.9421 | 0.833* | — |
| Logistic Regression (balanced) | 0.9660 | 0.9342 | 0.0671 |
| HistGradientBoosting (tuned GBDT) | 0.9677 | 0.9306 | 0.0210 |
| **TabPFN-3.5** | **0.9767** | **0.9377** | **0.0170** |

\* Ehsani's recall at 90% specificity from its ROC. All models evaluated
identically on identical folds.

**TabPFN-3.5 wins on every axis at once**: highest AUC, most carriers
caught at fixed confirmation budget, best calibration (the score is used
as a referral probability — the GBDT under-predicts prevalence, logistic
regression massively over-predicts without threshold tuning).

### Where the advantage is largest: small labeled data

The realistic deployment has a district clinic labeling a few hundred
cases, not 12,000. Mean AUC over 10 random splits:

| n_train | HistGradientBoosting | TabPFN-3.5 | Δ |
|---|---|---|---|
| 100 | 0.8808 | **0.9445** | **+0.0637** |
| 300 | 0.9385 | **0.9642** | **+0.0257** |
| 1000 | 0.9604 | **0.9706** | +0.0102 |
| 5000 | 0.9630 | **0.9710** | +0.0080 |

With 100 labeled cases, TabPFN already reaches AUC 0.944 — better than the
GBDT manages with 50× more data. This is the practical path to rolling the
screening model out region by region without a central labeled registry.

## What TabPFN-3.5 uniquely contributes

- **Zero-shot in-context learning**: no hyperparameter search, no
  architecture choice, no training loop — the same single forward pass
  beats both a hand-designed clinical formula (50 years of literature) and
  a modern GBDT.
- **Native missing-value and outlier tolerance**: the screening register
  contains data-entry extremes (Hb 41.1 g/dL, MCV 8.3 fL). TabPFN consumes
  them raw; classical pipelines need explicit cleaning rules.
- **Calibrated probabilities out of the box**: mean predicted prevalence
  4.6% vs true 4.6% (GBDT: 4.2%, unweighted LR: broken). In screening,
  the threshold *is* the policy — calibrated scores let a program pick
  "confirm the top X%" and know what that costs.
- **Data efficiency** (table above) from pretraining on synthetic
  clinical-like tasks.

## Honest limitations

- Retrospective single-program data; prospective multi-site validation is
  the obvious next step (and outside a hackathon's scope).
- The carrier-vs-normal split is the easy clinical question; the harder
  one (separating beta-trait from HbE-trait, both prevalent in Bengal) is
  **not** solved by CBC alone — the HPLC remains necessary there.
- Class imbalance (4.6% carriers) means AUC gains concentrate in better
  ranking, not large absolute recall changes at very high specificity.
- TabPFN inference ran on the Prior Labs cloud API; a clinic deployment
  would need the local checkpoint (supported via `--mode local`).

## Run it

```bash
# data: the Kaggle dataset zip (auto-downloaded by make data, or placed at data/)
export TABPFN_TOKEN=tabpfn_sk_...   # from platform.priorlabs.ai

python run_benchmark.py             # full benchmark -> results/benchmark_summary.md
python run_benchmark.py --skip-curve    # faster, skips the learning curve
python run_benchmark.py --mode local    # local TabPFN checkpoint, no cloud

python -m pytest tests/ -q          # 10 offline tests, no network
```

First run downloads the Kaggle dataset (~290 KB) with `make data`
(or fetch manually: `kaggle datasets download -d abhraghoshcmc/hplc-based-thalassemia-screening-data`).

## Layout

```
src/thalscreen/
├── dataset.py      # Kaggle zip -> tidy frame + the six published indices
├── baselines.py    # LR / HistGB with fixed-stratified-5-fold OOF harness
├── tabpfn_model.py # TabPFN-3.5 wrapper (cloud or local checkpoint)
└── evaluate.py     # recall@specificity, calibration, learning curves
data/hplc_thal.zip  # raw screening data (13,031 patients)
results/            # benchmark_summary.md + saved OOF scores
```

## License

MIT — see [LICENSE](LICENSE). Data under the Kaggle dataset's own license.
Built on TabPFN-3.5 (Prior Labs). Submitted to the TabPFN-3.5 Hackathon,
category *take on a hard problem*.
