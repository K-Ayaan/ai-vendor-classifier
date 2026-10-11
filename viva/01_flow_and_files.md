# 1. How Phase 1 flows, and which files to run first

## The problem in one sentence
Given a buyer's need ("I need a pharmacy within 5 km, preferably near a hospital"), score how
well each local vendor matches it. Phase 1 builds the statistical foundations for that with
two models from Unit II: **linear regression** (predict a 0–100 match score) and **logistic
regression** (classify a vendor as a pharmacy or not), both fitted by **maximum likelihood**.

## The data
23 JSON files of Google Maps place records (South Chennai to Chengalpattu, one snapshot on
2026-09-27): 5,439 records → 4,612 unique vendors after merging duplicates. There were no
labels, scores or requirements in the raw data, so Phase 1 had to **create** them.

## The flow (each box is one script)

```
data/raw/*.json  (never modified)
      │
      ▼
[0] src/audit.py ............. checks the raw files (counts, nulls, duplicates, pincodes)
      │                         → reports/audit.md
      ▼
[1] src/preprocessing.py ..... cleans + merges into one row per vendor, validates schema
      │                         → data/processed/vendors.csv (4,612 vendors)
      ├───────────────────────────────────────────────┐
      ▼                                               ▼
[2] src/make_requirements.py                  [3] src/logistic_regression_classification.py
    30 requirements × 15 candidates                is_pharmacy classifier (MLE, IRLS, L2)
    = 450 pairs + pair features                    → reports/classification_metrics.json
    → annotation_sheet.csv, RUBRIC.md              → label_audit_sample.csv (200 rows)
      │                                               │
      ▼ (pairs scored: AI-judged expected_score)      ▼ (audit labelled: AI-judged)
[4] src/linear_regression_mle.py              [3b] src/label_audit_rescore.py
    Gaussian MLE / least squares                    re-scores classifiers vs AI labels
    → reports/regression_metrics.json               → reports/label_audit_metrics.json
[4b] src/linear_regression_plots.py
    → reports/figures/*.png
      │                                               │
      └──────────────────┬────────────────────────────┘
                         ▼
[5] src/phase1_report.py ...... collects everything → reports/phase1_results.md + metrics.json
[6] src/extended_models.py .... ridge, Huber, Bayesian linear, kernel ridge, LDA, naive Bayes,
                                 Bayesian logistic (Laplace), SVMs → reports/extended_models_metrics.json
[7] src/make_viva_pack.py ..... graphs/01–19 + viva/00_key_numbers.md
notebooks/02_logistic_mle.ipynb, notebooks/03_linear_mle.ipynb ... theory + live results
```

## Which files to run, in order

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt     # pandas, numpy, scipy, scikit-learn, matplotlib, jupyter

python src/audit.py                            # 1. look at the raw data first
python src/preprocessing.py                    # 2. build the clean vendor table
python src/make_requirements.py                # 3. requirements + annotation pack (never overwrites scores)
python src/logistic_regression_classification.py   # 4. classifier
python src/label_audit_rescore.py              # 5. classifier vs AI-judged audit labels
python src/linear_regression_mle.py            # 6. regression
python src/linear_regression_plots.py          # 7. residual plots
python src/phase1_report.py                    # 8. final report
python src/extended_models.py                  # 9. extended Unit-II models
python src/make_viva_pack.py                   # 10. these viva graphs and numbers
jupyter notebook                               # 11. open notebooks/02 and 03 to explain the theory
```

**Why this order:** each script reads the previous one's output. You cannot build requirements
before the vendor table exists, and you cannot fit the regression before pairs are scored.
Every script is deterministic (seed 42): running the whole chain again from a fresh install
reproduced all 15 output files byte for byte.

## Rules the pipeline follows (examiners like these)
* **Raw data is read-only**: `data/raw/` is a checksum-verified copy of `Dataset/`.
* **No leakage**: vocabularies, pincode lists and the L2 strength are fitted on the
  training split only.
* **Honest splits**: classification uses a stratified 80/20 split; regression holds out
  whole requirements (24 train / 6 test) because in deployment the requirements are new.
* **Every number in the reports comes from code**, never typed by hand.
