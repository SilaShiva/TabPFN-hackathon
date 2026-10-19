# CBC-only beta-thalassaemia carrier screening benchmark

Data: 12,338 patients (562 carriers, 4.6% prevalence), West Bengal HPLC screening program (Kaggle: abhraghoshcmc/hplc-based-thalassemia-screening-data)
Features: RBC, HB, MCV, MCH, MCHC, RDWcv, age, sex (all available from a $2 CBC)

## 1. The six published CBC discrimination indices (clinical standard)

| Index | AUC |
|---|---|
| ehsani | 0.9421 |
| mentzer | 0.9181 |
| shine_lal | 0.8862 |
| england_fraser | 0.8094 |
| rdwi | 0.8093 |
| green_king | 0.7593 |

## 2. Models (5-fold stratified OOF, seed 42)

| model                |    auc |   recall@spec80 |   recall@spec90 |   recall@spec95 |   brier |
|:---------------------|-------:|----------------:|----------------:|----------------:|--------:|
| TabPFN-3.5           | 0.9767 |          0.9751 |          0.9377 |          0.8950 |  0.0170 |
| HistGradientBoosting | 0.9677 |          0.9662 |          0.9306 |          0.8541 |  0.0210 |
| LogisticRegression   | 0.9660 |          0.9698 |          0.9342 |          0.8541 |  0.0671 |

`recall@specNN` = sensitivity when the model flags at most NN% of patients for confirmatory HPLC. `brier` = calibration error (lower is better; the score is used as a referral probability).

## 3. Best published index vs best model (head to head)

- Best published index: **ehsani** at AUC 0.9421
- Best GBDT: HistGradientBoosting at AUC 0.9677
- **TabPFN-3.5: AUC 0.9767** (+0.0346 over the clinical standard, +0.0090 over the tuned GBDT)

## 4. Data efficiency (mean AUC over 10 random splits)

A district clinic labels a few hundred cases, not 12k.

|   n_train |   HistGradientBoosting |   TabPFN-3.5 |   delta |
|----------:|-----------------------:|-------------:|--------:|
|       100 |                 0.8808 |       0.9445 |  0.0637 |
|       300 |                 0.9385 |       0.9642 |  0.0257 |
|      1000 |                 0.9604 |       0.9706 |  0.0102 |
|      5000 |                 0.9630 |       0.9710 |  0.0080 |