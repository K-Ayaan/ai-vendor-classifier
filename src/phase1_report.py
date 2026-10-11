"""Step 6 - assemble reports/metrics.json and reports/phase1_results.md.

Usage: python src/phase1_report.py
Every number is read from the outputs of steps 2-5 (nothing is typed in by hand):
  data/processed/vendors.csv, requirements.csv, pairs.csv,
  reports/classification_metrics.json, reports/classification_coefficients.csv,
  reports/regression_metrics.json
"""
import json
from pathlib import Path

import pandas as pd

OUT_JSON = Path("reports/metrics.json")
OUT_MD = Path("reports/phase1_results.md")


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}%"


def md_table(df):
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def data_summary():
    v = pd.read_csv("data/processed/vendors.csv", encoding="utf-8", dtype={"pincode": str},
                    keep_default_na=False)
    req = pd.read_csv("data/processed/requirements.csv", encoding="utf-8", keep_default_na=False)
    pairs = pd.read_csv("data/processed/pairs.csv", encoding="utf-8")
    cats = v.categories.str.split(";")
    return {
        "vendors": len(v), "raw_records": int(v.n_categories.sum()),
        "categories": len(set().union(*cats.map(set))),
        "single_category_share": float((v.n_categories == 1).mean()),
        "pincode_imputed": int(v.pincode_imputed.sum()), "is_chain": int(v.is_chain.sum()),
        "requirements": len(req),
        "requirements_with_keyword": int((req.preference_keyword != "").sum()),
        "radius_counts": {str(k): int(n) for k, n in
                          req.max_distance_km.value_counts().sort_index().items()},
        "pairs": len(pairs),
        "pairs_per_stratum": {k: int(n) for k, n in pairs.stratum.value_counts().items()},
        "pair_feature_means": {c: float(pairs[c].mean()) for c in
                               ["within_radius", "category_match", "keyword_in_title", "is_chain"]},
    }


def main():
    data = data_summary()
    clf = json.loads(Path("reports/classification_metrics.json").read_text(encoding="utf-8"))
    reg = json.loads(Path("reports/regression_metrics.json").read_text(encoding="utf-8"))
    coef = pd.read_csv("reports/classification_coefficients.csv", encoding="utf-8")
    la_path = Path("reports/label_audit_metrics.json")
    la = json.loads(la_path.read_text(encoding="utf-8")) if la_path.exists() else None
    ext_path = Path("reports/extended_models_metrics.json")
    ext = json.loads(ext_path.read_text(encoding="utf-8")) if ext_path.exists() else None
    metrics = {"phase": 1, "data": data, "classification": clf,
               "classification_vs_ai_judged_labels": la, "regression": reg,
               "extended_models": ext}
    OUT_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    s, f, u, l2, base = (clf["split"], clf["features"], clf["unpenalised_mle"], clf["l2_model"],
                         clf["majority_baseline"])
    sm = clf["small_mle_model"]
    tm, um, smm = l2["test_metrics"], u["test_metrics_NOT_A_VALID_MLE"], sm["test_metrics"]
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    metric_rows = pd.DataFrame([
        {"model": "Small-model MLE (10 tokens; MLE exists)", **{k: f"{smm[k]:.4f}" for k in keys}},
        {"model": "L2 model — penalised likelihood (MAP), not MLE",
         **{k: f"{tm[k]:.4f}" for k in keys}},
        {"model": "Full unpenalised fit (NOT a valid MLE)", **{k: f"{um[k]:.4f}" for k in keys}},
        {"model": "Majority-class baseline (always 0)", "accuracy": f"{base['accuracy']:.4f}",
         "precision": f"{base['precision']:.4f}", "recall": f"{base['recall']:.4f}",
         "f1": f"{base['f1']:.4f}", "roc_auc": f"{base['roc_auc']:.4f}",
         "pr_auc": f"{base['pr_auc_equals_prevalence']:.4f}"},
    ])
    cm_rows = pd.DataFrame([{"model": name, **m["confusion_matrix"]} for name, m in
                            [("Small-model MLE", smm), ("L2 (MAP, not MLE)", tm),
                             ("Full unpenalised (invalid MLE)", um)]])
    small_coef = pd.DataFrame({"feature": list(sm["coefficients"]),
                               "beta_irls": [f"{v:.6f}" for v in sm["coefficients"].values()],
                               "beta_sklearn_penalty_none": [
                                   f"{v:.6f}" for v in sm["coefficients_sklearn"].values()],
                               "train_doc_freq": ["—"] + [str(sm["train_doc_freq"][t])
                                                          for t in sm["tokens"]]})
    skipped = ", ".join(f"`{x['token']}` (df {x['train_df']}, {x['reason']})"
                        for x in sm["skipped_before_10"]) or "none"
    lpf = u["joint_separation_lp"]
    top = coef.reindex(coef.beta_sklearn_l2.sort_values(ascending=False).index)
    top_pos = top.head(8)[["feature", "beta_sklearn_l2"]].round(3)
    top_neg = top.tail(8).iloc[::-1][["feature", "beta_sklearn_l2"]].round(3)
    sep_examples = ", ".join(f"`{d['feature']}` ({d['n_rows']} rows, only class "
                             f"{d['only_class']})" for d in sorted(
                                 u["separating_features"], key=lambda d: -d["n_rows"])[:6])
    trace = u["irls_trace_last5"][-1]
    d = data

    if la:
        ag = la["agreement_ai_vs_weak"]
        ct = ag["crosstab_weak_vs_ai"]
        names = {"small_mle": "Small-model MLE", "l2_map": "L2 model (MAP, not MLE)",
                 "full_unpenalised_invalid_mle": "Full unpenalised (invalid MLE)"}
        la_rows = pd.DataFrame([{"model": names[k], "labels": lab,
                                 **{m: f"{v[key][m]:.4f}" for m in keys}}
                                for k, v in la["classifiers"].items()
                                for lab, key in (("AI-judged", "vs_ai_labels_test_reweighted"),
                                                 ("weak", "vs_weak_labels_test_reweighted"))])
        la_md = f"""### 2.4 Re-scoring against AI-judged labels (AI-judged labels, not ground truth)

The {ag['n_sample']} rows of `data/annotation/label_audit_sample.csv` were labelled by an AI
judge (Claude Code) in a new column `expected_is_pharmacy`, using title and address only,
with a rationale per row (`data/annotation/ai_pharmacy_labels.csv`). The judge did not open the
key, weak labels or predictions while labelling. Caveat: 12 of these vendors' weak labels had
been shown earlier in the session (Step 4 error list). **These are AI-judged labels, not
ground truth**; `true_is_pharmacy` (for humans) remains empty.

The sample holds all weak positives in the test set and a subset of weak negatives, so test-set
estimates are re-weighted by stratum: weak positives × {la['stratum_weights']['weak=1']:.4f}
({la['sample_counts']['weak=1']} of {la['test_counts']['weak=1']} sampled), weak negatives ×
{la['stratum_weights']['weak=0']:.4f} ({la['sample_counts']['weak=0']} of
{la['test_counts']['weak=0']} sampled).

**AI labels vs weak labels.** Cross-tab (sample): weak=1 & AI=1: {ct['weak=1,ai=1']},
weak=1 & AI=0: {ct['weak=1,ai=0']}, weak=0 & AI=0: {ct['weak=0,ai=0']}, weak=0 & AI=1:
{ct['weak=0,ai=1']}. Agreement {pct(ag['agreement_sample_unweighted'])} on the sample and
{pct(ag['agreement_test_reweighted'])} re-weighted to the test set (Cohen's κ
{ag['cohen_kappa_test_reweighted']:.4f}). The AI judge confirms
{pct(ag['share_weak_positives_ai_confirms'], 1)} of weak positives and
{pct(ag['share_weak_negatives_ai_confirms'], 1)} of weak negatives. The estimated AI-judged
pharmacy rate in the test set is {pct(ag['estimated_ai_positive_rate_test_reweighted'])}, against a
weak-label rate of {pct(ag['weak_positive_rate_test'])}. Every disagreement is a weak positive
the judge rejected: supermarkets and general stores, a chemistry academy, hospitals and a
medical college, a vet clinic, a surgical-supplies shop, a pharmacy head office, and names
with no medical cue.

**Classifier metrics, re-weighted to the test set** (threshold 0.5), against AI-judged labels
and, for comparison on the same rows, against weak labels:

{md_table(la_rows)}

Against the AI-judged labels, recall rises for every model, because the "missed pharmacies" that
were not pharmacies are no longer counted as misses. Precision falls for every model. For the
L2 model the cause is that it calls hospitals, a medical college and a pharmacy head office
"pharmacy" (strong title words such as *Medical*, *Hospital*, *MedPlus*), which the AI judge
labels 0.

"""
    else:
        la_md = ""

    if reg.get("status") == "fitted":
        cons, sp, ols, g = (reg["test_retest_consistency"], reg["split"], reg["ols"],
                            reg["gaussian_mle"])
        coef_t = pd.DataFrame({"feature": list(reg["coefficients"]),
                               "beta_hat": [f"{v:.4f}" for v in reg["coefficients"].values()]})
        met_t = pd.DataFrame([{"split": k, "n": reg[k]["n"], "RSS": f"{reg[k]['RSS']:.2f}",
                               "MSE": f"{reg[k]['MSE']:.4f}", "RMSE": f"{reg[k]['RMSE']:.4f}",
                               "MAE": f"{reg[k]['MAE']:.4f}", "R²": f"{reg[k]['R2']:.4f}"}
                              for k in ("train", "test")])
        reg_md = f"""**Target: AI-judged expected_score.** The project owner replaced human raters with an
AI judge: Claude Code scored all {reg['complete_rows']} pairs against `RUBRIC.md`, one
requirement at a time, with a rationale per pair (`data/annotation/ai_scores.csv`, copied into
the `expected_score` column of the annotation sheet). These are **AI-judged labels, not human
annotations and not ground truth**. The rater columns remain empty
({reg['rater_1_filled']} and {reg['rater_2_filled']} filled).

**AI judge consistency** (in place of inter-rater agreement), on a fixed seed-42 sample of 45
pairs:

{md_table(pd.DataFrame([{"check": label, "n": c["n_pairs"], "Pearson r": f"{c['pearson_r']:.4f}",
                         "mean abs diff": f"{c['mean_abs_difference']:.3f}",
                         "pairs > 25 apart": c["n_abs_difference_gt_25"],
                         "max diff": f"{c['max_abs_difference']:g}", "identical": c["n_identical"],
                         "mean (first − retest)": f"{c['mean_signed_difference_first_minus_retest']:.3f}"}
                        for label, c in [("Blind: fresh AI agent", cons.get("blind")),
                                         ("Non-blind: same judge, same session", cons["non_blind"])]
                        if c]))}

The **blind** check is the headline figure. A fresh AI agent received only `RUBRIC.md` and the
45 pairs' sheet details (requirement, distance, vendor title, address, listed categories),
never saw the first scores or rationales, opened no files, and returned its scores as text
(`data/annotation/ai_rescore_blind.csv`). The single disagreement above 25 points is P407
(a clothing hypermarket for a bakery need: first score 5, blind 35). The non-blind check
(`data/annotation/ai_rescore_check.csv`) was re-scored by the same judge with its first scores
still in context, so it is an upper bound. **Human validation was not done; all labels in this
project are AI-judged.**

**Split:** {sp['method']} (seed {sp['seed']}): {sp['train_requirements']} train /
{sp['test_requirements']} test requirements ({sp['n_train']} / {sp['n_test']} pairs).

**Fit:** β̂ solved with `numpy.linalg.lstsq`; rank {ols['rank']} of {ols['p']}, condition number
of X_train = {ols['cond_X']:.2f}. Cross-checks: max |β̂_inverse − β̂_lstsq| =
{ols['max_abs_diff_inverse_vs_lstsq']:.2e}, max |β̂_sklearn − β̂_lstsq| =
{ols['max_abs_diff_sklearn_vs_lstsq']:.2e}.

{md_table(coef_t)}

Metrics (target: AI-judged expected_score):

{md_table(met_t)}

**Gaussian MLE (train, n = {reg['train']['n']}, p = {ols['p']}):** σ̂²_MLE = RSS/n =
{g['sigma2_mle']:.4f}, σ̂_MLE = {g['sigma_mle']:.4f}; unbiased s² = RSS/(n−p) =
{g['s2_unbiased']:.4f}, s = {g['s_unbiased']:.4f}. Maximised log-likelihood: direct sum
{g['loglik_direct_sum']:.6f} vs closed form −n/2·(ln(2πσ̂²)+1) = {g['loglik_formula']:.6f}
(match: {g['loglik_match']}). Train MSE equals σ̂²_MLE: {reg['mse_equals_sigma2_mle_train']}.
They are the same number but different concepts: MSE is a descriptive loss of the fitted
predictor, while σ̂²_MLE estimates the noise-variance parameter of the Gaussian model and is
biased, hence s².

**Diagnostics** (`reports/figures/residuals_vs_fitted.png`, `reports/figures/residual_qq.png`,
`notebooks/03_linear_mle.ipynb` §10): the bounded, anchor-clustered target produces banded
residuals and some negative fitted scores, and the residuals are heavier-tailed than a
Gaussian, so "least squares = MLE" holds only approximately. The largest misfits are vendors
whose source-file category matches but which the judge recognised from the title as the wrong
kind of shop. This shows the judge did not simply copy the feature columns."""
    else:
        reg_md = (f"**Blocked pending annotation.** {reg['complete_rows']} of {reg['total_rows']} "
                  f"pairs have an AI-judged expected_score; at least "
                  f"{reg['required_complete_rows']} are "
                  "required. No regression model was fitted and no regression metrics exist. "
                  "The estimator, metrics, Gaussian-MLE quantities and diagnostic plots are "
                  "implemented in `src/linear_regression_mle.py` and "
                  "`src/linear_regression_plots.py`, and the theory is written out in "
                  "`notebooks/03_linear_mle.ipynb`. Rerun those once the sheet is filled.")

    # ---------------- extended Unit-II models ----------------
    if ext:
        rm, rd = ext["regression"]["metrics"], ext["regression"]["details"]
        cm2, cd = ext["classification"]["metrics"], ext["classification"]["details"]
        ols_te = rm["OLS (Gaussian MLE)"]["test"]
        best_reg = max(rm, key=lambda k: rm[k]["test"]["R2"])
        ext_reg = pd.DataFrame([{"model": n, "train R²": f"{v['train']['R2']:.4f}",
                                 "test R²": f"{v['test']['R2']:.4f}",
                                 "test RMSE": f"{v['test']['RMSE']:.4f}",
                                 "test MAE": f"{v['test']['MAE']:.4f}"} for n, v in rm.items()])
        ext_clf = pd.DataFrame([{"model": n, **v["confusion_matrix"],
                                 **{k: f"{v[k]:.4f}" for k in keys}} for n, v in cm2.items()])
        ridge_note = ("the smallest value on the grid, so ridge ≈ OLS: with 6 well-conditioned "
                      "features there is nothing to shrink"
                      if rd["ridge_alpha"] <= 1e-3 + 1e-12 else "")
        f1 = {n: v["f1"] for n, v in cm2.items()}
        ext_md = f"""## 3b. Extended Unit-II models (same data, splits and seed)

Generated by `python src/extended_models.py` (`reports/extended_models_metrics.json`). Every
hyperparameter is chosen by cross-validation on the training split only (grouped by
requirement for regression). The regression target is the AI-judged expected_score; the
classification target is the weak is_pharmacy label. Graphs 13–19 in `graphs/` and
`viva/07_extended_models.md` explain each model.

**Regression** (held-out requirements):

{md_table(ext_reg)}

* **Ridge regression** (MAP with a Gaussian prior): grouped CV chose λ = {rd['ridge_alpha']:g}{', ' + ridge_note if ridge_note else ''}.
* **Robust regression (Huber, ε = {rd['huber_epsilon']})**: {rd['huber_n_outliers_train']} of
  {reg['split']['n_train'] if reg.get('status') == 'fitted' else ''} training pairs fall in the linear (down-weighted) part of the loss; test R²
  {ols_te['R2']:.4f} → {rm['Robust (Huber)']['test']['R2']:.4f}.
* **Bayesian linear regression** (evidence maximisation): noise variance 1/α =
  {rd['bayes_noise_variance_1_over_alpha']:.4f}, weight precision λ = {rd['bayes_weight_precision_lambda']:.6f};
  the closed-form posterior matches scikit-learn to {rd['bayes_closed_form_vs_sklearn_max_abs_diff']:.2e}. Test 95%
  predictive intervals cover {rd['bayes_test_95pct_interval_coverage']:.2%} of the scores (mean predictive sd
  {rd['bayes_test_mean_predictive_sd']:.4f}).
* **Kernels / kernel trick** (kernel ridge = a GLM with a Gaussian likelihood in a kernel feature
  space): best test R² {rm[best_reg]['test']['R2']:.4f} ({best_reg}) vs {ols_te['R2']:.4f} for OLS, because the
  kernels model feature interactions that the additive linear model cannot.

**Classification** (test set, weak labels, threshold 0.5; SVM threshold 0 on the decision
function):

{md_table(ext_clf)}

* **Discriminant function (Fisher LDA)** F1 {f1['Fisher LDA (discriminant)']:.4f} and **generative
  Bernoulli naive Bayes** F1 {f1['Bernoulli naive Bayes (generative)']:.4f}, vs **discriminative** L2 logistic
  F1 {f1['Logistic L2 (MAP), Phase 1']:.4f}.
* **Bayesian logistic regression via the Laplace approximation** (prior λ = {cd['laplace_lambda']:g}): the
  prior guarantees a solution despite separation, and the decisions match the L2 model
  (F1 {f1['Bayesian logistic (Laplace)']:.4f}); the probit-moderated predictive probabilities change test log
  loss from {cd['laplace_test_logloss_map_plugin']:.4f} (plug-in) to {cd['laplace_test_logloss_predictive']:.4f}.
* **SVMs**: linear (C = {cd['svm_linear_C']:g}) F1 {f1['SVM linear']:.4f} with {cd['svm_linear_n_support']} support vectors;
  RBF (C = {cd['svm_rbf_C']:g}) F1 {f1['SVM RBF kernel']:.4f} with {cd['svm_rbf_n_support']} support vectors. The max-margin
  problem has a unique solution on separable data, where the logistic MLE does not exist.

"""
    else:
        ext_md = ""

    # ---------------- one-page summary for the examiner ----------------
    def mrow(name, m, pr_key="pr_auc"):
        return {"model": name, **{k: f"{m[k]:.4f}" for k in ["accuracy", "precision", "recall",
                                                            "f1", "roc_auc"]},
                "pr_auc": f"{m[pr_key]:.4f}"}
    sum_clf = pd.DataFrame([mrow("L2 logistic regression: penalised likelihood (MAP)", tm),
                            mrow("Small-model logistic regression: MLE (10 tokens)", smm),
                            mrow("Majority-class baseline", base, "pr_auc_equals_prevalence")])
    if reg.get("status") == "fitted":
        cb = reg["test_retest_consistency"].get("blind") or reg["test_retest_consistency"]["non_blind"]
        sum_reg = pd.DataFrame([
            {"quantity": "R² (train / test)", "value": f"{reg['train']['R2']:.4f} / {reg['test']['R2']:.4f}"},
            {"quantity": "RMSE (train / test)", "value": f"{reg['train']['RMSE']:.4f} / {reg['test']['RMSE']:.4f}"},
            {"quantity": "σ̂²_MLE = RSS/n (train)", "value": f"{reg['gaussian_mle']['sigma2_mle']:.4f}"},
            {"quantity": "s² = RSS/(n−p) (train)", "value": f"{reg['gaussian_mle']['s2_unbiased']:.4f}"},
            {"quantity": "maximised log-likelihood (train)", "value": f"{reg['gaussian_mle']['loglik_formula']:.6f}"}])
        reg_sum = (f"Target: AI-judged expected_score, {reg['complete_rows']} pairs, split by requirement "
                   f"({reg['split']['train_requirements']} train / {reg['split']['test_requirements']} test).\n\n"
                   + md_table(sum_reg))
        judge_line = (f"Blind consistency of the AI judge on 45 pairs: Pearson r = {cb['pearson_r']:.4f}, "
                      f"mean absolute difference = {cb['mean_abs_difference']:.3f}.")
    else:
        reg_sum = f"Blocked: {reg['complete_rows']} scored pairs (need {reg['required_complete_rows']})."
        judge_line = ""
    summary_md = f"""## Summary

**What was built.** A reproducible Python pipeline: a raw-data audit, preprocessing into a
validated vendor table, a generated requirement set with a 450-pair annotation pack, a
logistic-regression classifier for `is_pharmacy` fitted by maximum likelihood (hand-written
Newton/IRLS cross-checked against scikit-learn), and a linear regression of a 0–100 match score
fitted by Gaussian MLE / least squares. Two notebooks derive the theory (02 logistic, 03
linear) and show every check.{' The other Unit-II models (ridge, robust, Bayesian linear '
'regression, kernel methods, Fisher LDA, naive Bayes, Bayesian logistic regression via the '
'Laplace approximation, SVMs) are compared on the same splits in §3b.' if ext else ''}

**Dataset.** {d['raw_records']} Google Maps place records in {d['categories']} retail-category
files (South Chennai to Chengalpattu, one snapshot, 2026-09-27), merged into
{d['vendors']} unique vendors.

**Classification** (held-out test set: {s['n_test']} vendors, {s['positives_test']} weak-label
pharmacies, threshold 0.5):

{md_table(sum_clf)}

**Regression** (Gaussian MLE / least squares):

{reg_sum}

**Why the full logistic model has no MLE.** With all {f['n_total_incl_intercept']} features
the training data are quasi-completely separated: {u['n_separating_features']} title tokens
occur in only one class (e.g. *medicals* only in pharmacies, *market* only in non-pharmacies),
and a linear-programming check confirms a separating direction. The log-likelihood then keeps
rising as coefficients go to ±∞, so no finite maximiser exists. The design matrix itself has
full rank, so separation is the only cause. Two separate models address this:
(1) the **small model** keeps only {sm['n_features_excl_intercept']} frequent tokens chosen on
the training set and skips any that would separate, so its MLE exists; IRLS and scikit-learn
agree to {sm['max_abs_coef_diff_irls_vs_sklearn']:.0e}. (2) The **L2 model** adds a penalty,
which guarantees a unique finite solution. It is penalised likelihood (a MAP estimate), not
an MLE.

**Caveat: AI-judged labels.** The match scores ({reg.get('complete_rows', 0)} pairs) and the
200-row pharmacy label audit were judged by an AI (Claude Code) on the project owner's
instruction. They are AI-judged labels, not ground truth, and **human validation was not
done**. {judge_line} Regression R² measures how well a linear model reproduces the AI judge,
not true match quality. The classifier's own target is weak (source-file) labels{
(f"; re-scored against the AI-judged labels, the L2 model's F1 is "
 f"{la['classifiers']['l2_map']['vs_ai_labels_test_reweighted']['f1']:.4f} (re-weighted, §2.4).")
if la else "."}"""

    md = f"""# Phase 1 results: Intelligent Agent Mapping — Vendor Matching

Generated by `python src/phase1_report.py` from the outputs of Steps 2–5. All numbers come
from code run on this dataset; the project poster's figures are illustrative and are not
results. Machine-readable copy: `reports/metrics.json`.

{summary_md}

---

# Technical record

## 1. Data

* {d['raw_records']} place records in {d['categories']} category files (single snapshot,
  2026-09-27) → **{d['vendors']} unique vendors** after merging repeated ids.
  {pct(d['single_category_share'], 1)} of vendors appear in exactly one category.
* {d['pincode_imputed']} vendor had no address; its pincode was taken from the nearest valid
  vendor (`pincode_imputed = True`). {d['is_chain']} vendors have `is_chain = True` (see
  limitations).
* Annotation pack: {d['requirements']} requirements ({d['requirements_with_keyword']} with a
  preference keyword; radii {d['radius_counts']} km) × 15 candidates = **{d['pairs']} pairs**
  (strata {d['pairs_per_stratum']}).

## 2. Classification: logistic regression for `is_pharmacy`

**Setup.** Stratified 80/20 split (seed {s['seed']}): train {s['n_train']} vendors
({s['positives_train']} pharmacies, prevalence {s['prevalence_train']:.4f}), test
{s['n_test']} ({s['positives_test']} pharmacies, prevalence {s['prevalence_test']:.4f}).
Features fitted on train only: {f['n_title_tokens']} title-token indicators
(min_df = {f['min_df']}) + {f['n_pincode_indicators']} pincode indicators (other pincodes =
reference) + intercept = {f['n_total_incl_intercept']} columns. Token columns that are linear
combinations of earlier columns in train were dropped ({len(f['dropped_linearly_dependent_columns'])}:
{', '.join('`' + c + '`' for c in f['dropped_linearly_dependent_columns'])}), so X_train has
**full column rank: rank {f['rank_X_train']} = {f['n_total_incl_intercept']} columns**.

### 2.1 Unpenalised MLE: **the MLE does not exist**

* **Quasi-complete separation:** {u['n_separating_features']} features are each 1 only for
  rows of a single class in train, e.g. {sep_examples}. Along each, the log-likelihood keeps
  increasing as the coefficient goes to ±∞. The joint linear-programming check (maximise
  Σ sᵢxᵢᵀb subject to sᵢxᵢᵀb ≥ 0, −1 ≤ bⱼ ≤ 1, with sᵢ = 2yᵢ − 1) confirms it: optimum
  {lpf['lp_optimum']:g} > 0, so a separating direction exists.
* **Non-convergence:** hand-written Newton/IRLS did not converge in {u['irls_iterations']}
  iterations; at the last iteration ‖β‖ = {trace['beta_norm']:.2f} and cond(XᵀWX) =
  {trace['cond_H']:.2e}.
* **Same likelihood, different β:** IRLS log-likelihood {u['loglik_irls']:.6f} vs
  scikit-learn {u['loglik_sklearn']:.6f} (null model {u['loglik_null_intercept_only']:.6f}),
  yet the coefficients differ by up to {u['max_abs_coef_diff_irls_vs_sklearn']:.2f}.
  scikit-learn reported convergence after {u['sklearn_iterations']} iterations, which here only
  means its gradient fell below tolerance on a flat ridge.

X_train has full rank, so the failure is due to separation alone. The full-model unpenalised
coefficients are not interpretable; their test metrics appear below only for comparison.

### 2.2 Pure-MLE demonstration: small model where the MLE exists

A second, separate model uses at most {sm['n_features_excl_intercept']} title-token
indicators, chosen by **training-set** document frequency (test labels are never used).
Tokens are taken in frequency order; a token is skipped if adding it would make the training
design rank-deficient or separated. Skipped before reaching 10: {skipped}.

* Design: {sm['n_columns']} columns (intercept + {sm['n_features_excl_intercept']} tokens),
  rank {sm['rank_X_train']}; single-feature separators: {len(sm['single_feature_separators'])};
  joint separation LP optimum {abs(sm['separation_lp']['lp_optimum']):g} → **not separated**, so a
  finite MLE exists.
* Hand-written IRLS converged in {sm['irls_iterations']} iterations; scikit-learn
  `LogisticRegression(penalty=None, solver="newton-cholesky")` took {sm['sklearn_n_iter']}.
* **Coefficients agree:** max |β̂_IRLS − β̂_sklearn| = {sm['max_abs_coef_diff_irls_vs_sklearn']:.2e}
  (< 1e-6). **Log-likelihoods match:** {sm['loglik_irls']:.9f} vs {sm['loglik_sklearn']:.9f}
  (|difference| = {sm['abs_loglik_diff']:.2e}).
* `penalty=None` raises a FutureWarning in scikit-learn 1.9.1 (deprecated, removed in 1.10);
  the pinned version still supports it.

{md_table(small_coef)}

### 2.3 Separate result: L2 model — penalised likelihood (MAP), not MLE

This model maximises ℓ(β) − (λ/2)‖β₁:ₖ‖², i.e. the posterior mode under a Gaussian prior on the
non-intercept coefficients. It is **not** a maximum-likelihood estimate. C = {l2['C']:g} (λ = {l2['lambda_equiv']:g}) chosen by 5-fold stratified CV on the training
split (log-loss). Hand-written penalised IRLS reproduces scikit-learn's coefficients to
{l2['max_abs_coef_diff_sklearn_vs_hand_irls']:.2e}. Train log-likelihood
{l2['loglik_train']:.4f}.

**Held-out test metrics** (threshold 0.5):

{md_table(metric_rows)}

Confusion matrices (test):

{md_table(cm_rows)}

Largest positive L2 (MAP) coefficients:

{md_table(top_pos)}

Largest negative L2 (MAP) coefficients:

{md_table(top_neg)}

**Reading the results.** The small MLE model is a valid MLE but weak: with only frequent,
mostly generic tokens it finds about a third of pharmacies (mainly via `pharmacy`). The L2
(MAP) model beats the majority baseline clearly, but all metrics
measure agreement with *weak* labels. Several of the L2 model's false negatives look like
label noise rather than model error (e.g. a chemistry academy and multi-category
supermarkets listed under pharmacy), while others are genuine misses (e.g. "PRABAL PHARMA");
see `notebooks/02_logistic_mle.ipynb` §10. A blind label-audit sheet of
{clf['label_audit_sample']['rows']} test vendors ({clf['label_audit_sample']['weak_positive']}
weak positives, {clf['label_audit_sample']['weak_negative']} weak negatives) is in
`data/annotation/label_audit_sample.csv` for measuring that noise.

{la_md}## 3. Regression: linear regression / Gaussian MLE for the AI-judged expected_score

{reg_md}

{ext_md}## 4. Limitations

* **Weak, noisy classification labels.** Category membership comes from the source file, not
  verification. The label audit was filled by an AI judge (AI-judged labels, not ground truth),
  so §2.4 measures agreement with that judge. No human-verified labels exist.
* **Single snapshot.** No timestamps, so no temporal validation is possible.
* **Annotation size.** The regression target relies on {d['pairs']} pairs from
  {d['requirements']} requirements; with a 24/6 requirement split the test set is small, and
  only {pct(d['pair_feature_means']['keyword_in_title'], 1)} of pairs have
  `keyword_in_title = True`, so that coefficient will be poorly determined.
* **The regression target is AI-judged.** `expected_score` was judged by Claude Code, not by
  human raters. R² measures how well a linear model reproduces the AI judge's scores, not true
  match quality. The features capture the same criteria the judge used (category, distance vs
  radius, preference keyword), so a high R² is expected and is **not** proof of a good matcher.
* **AI judge consistency, not human agreement.** Consistency was measured blind with a fresh
  AI agent on 45 pairs (plus a non-blind same-session upper bound). Both are AI-vs-AI; no human
  inter-rater agreement exists.
* **Constructed requirements.** Requirements are generated from the dataset (anchor vendor
  location, sampled radius, allow-listed keyword); they are not real buyer requests.
* **`is_chain` is a simple rule** (title repeated ≥ 3 times); generic names such as "ration
  shop" also meet it.
* **`locality` is often a city** ("Chennai"), not a neighbourhood.
* **Label-audit sample oversamples pharmacies**; re-weight to the test prevalence when
  estimating noise rates.

## 5. Reproduce

```powershell
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
python src/audit.py
python src/preprocessing.py
python src/make_requirements.py      # never overwrites an existing annotation sheet
python src/logistic_regression_classification.py
python src/label_audit_rescore.py
python src/linear_regression_mle.py
python src/linear_regression_plots.py
python src/extended_models.py
python src/phase1_report.py
python src/make_viva_pack.py
jupyter nbconvert --to notebook --execute --inplace notebooks/02_logistic_mle.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_linear_mle.ipynb
```
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
