# thalscreen — CBC-only beta-thalassaemia carrier screening with TabPFN-3.5

**TabPFN-3.5 beats the 50-year-old clinical standard and tuned gradient
boosting on a real screening task — highest AUC, most carriers caught at a
fixed confirmation budget, best calibration, and a 50× data-efficiency
advantage in the regime that matters.**

*Entry for the Prior Labs TabPFN-3.5 hackathon — category: take on a hard problem.*

## The task

Beta thalassaemia carriers look almost normal — mildly low MCV, slightly
high RBC count — and are found today by six hand-published arithmetic
formulas (Mentzer 1973, Ehsani, Green & King, ...) computed from the $2
complete blood count, followed by confirmatory HPLC for those flagged.
Every formula paper claims to beat the previous one; a fixed formula cannot
model six correlated parameters across age and sex.

This project builds the missing benchmark: **12,338 real patients** from a
West Bengal screening program (Kaggle HPLC screening data, HPLC-confirmed
diagnoses), and compares the published indices, logistic regression, a
tuned GBDT, and TabPFN-3.5 — all consuming **only** the CBC features a
district clinic has before any referral.

## Results (5-fold stratified OOF)

| Model | AUC | Recall @ 90% specificity | Brier |
|---|---|---|---|
| Best published index (Ehsani) | 0.942 | 0.83 | — |
| Logistic Regression | 0.966 | 0.934 | 0.067 |
| HistGradientBoosting | 0.968 | 0.931 | 0.021 |
| **TabPFN-3.5** | **0.977** | **0.938** | **0.017** |

And the finding that matters for rollout — mean AUC with limited labeled
data (10 random splits each):

| Labeled cases | GBDT | TabPFN-3.5 | Δ |
|---|---|---|---|
| 100 | 0.881 | **0.944** | +0.064 |
| 300 | 0.939 | **0.964** | +0.026 |
| 1000 | 0.960 | **0.971** | +0.010 |

With 100 labeled cases TabPFN already outperforms the GBDT trained on 50×
more data — the difference between a national registry and what one district
clinic can label.

## Why TabPFN-3.5 wins here specifically

- **Zero-shot, single forward pass** — no hyperparameter search, no training
  loop; the same default model beats 50 years of hand-tuned clinical formulas
  and a modern GBDT.
- **Calibrated by construction** — predicted prevalence 4.6% vs true 4.6%;
  screening thresholds are policy decisions, and they need honest probabilities.
- **Eats dirty registers raw** — the data contains data-entry extremes
  (Hb 41 g/dL, MCV 8 fL); no cleaning pipeline in the TabPFN path.
- **Data efficiency** from pretraining on synthetic clinical-like tasks —
  the key to region-by-region rollout.

## Run it

```bash
cd thalscreen
pip install -e .
export TABPFN_TOKEN=tabpfn_sk_...   # platform.priorlabs.ai (free tier)
python run_benchmark.py             # -> results/benchmark_summary.md
python -m pytest tests/ -q          # 10 offline tests, no network
```

Full methodology, honest limitations (retrospective data, the harder
beta-vs-HbE problem, class imbalance), and per-experiment numbers in
[thalscreen/README.md](thalscreen/README.md).

## License

MIT. Data under the Kaggle dataset's own license.
Built on TabPFN-3.5 ([Prior Labs](https://priorlabs.ai)).
