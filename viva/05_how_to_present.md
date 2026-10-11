# 5. How to explain Phase 1 to someone, step by step (about 10–12 minutes)

## Step 1: The problem (1 min)
"Buyers need local vendors. Given a need, a location, a distance limit and a preference, we
want to score and classify vendors. Phase 1 builds the statistical core with two models from
Unit II, both fitted by maximum likelihood: linear regression for a match score, and logistic
regression for a category."

## Step 2: The data, and what was missing (1 min)
"5,439 Google Maps records from 23 category files, merged into 4,612 vendors. There were no
labels, scores or requirements, so we created them: 30 requirements × 15 candidates = 450
pairs. The scores are AI-judged labels, not ground truth. The category label comes from which
file a vendor is in, so it is noisy." Show `reports/audit.md` if asked.

## Step 3: The pipeline (1 min)
Walk through the flow diagram in `01_flow_and_files.md`: audit → preprocessing → requirements
→ classifier → regression → report. Mention raw data is read-only, the seed is 42, and every
number is produced by code.

## Step 4: Features (1 min)
Use the table in `02_features.md`. One sentence each for the six regression features, then
"for classification: 421 title words, 20 pincodes, plus an intercept, all fitted on the
training split only".

## Step 5: Linear regression = Gaussian MLE (3 min), the core of Unit II
1. Write y = Xβ + ε, ε ~ N(0, σ²).
2. Write ℓ = −(n/2)ln(2πσ²) − RSS/(2σ²).
3. Say **why minimising RSS = maximising ℓ** (β only appears in −RSS/(2σ²)).
4. β̂ = (XᵀX)⁻¹Xᵀy, solved stably with lstsq (κ(X) = 114.45).
5. Show graph `05` (coefficients): category match +59.44, inside radius +18.14, keyword
   +15.47.
6. Show graph `06` (ℓ vs σ²): the peak is at σ̂²_MLE = RSS/n = 224.28; s² = 228.72 is the
   unbiased version; ℓ_max = −1485.137.
7. Show graph `07`: R² train 0.7983, test 0.7510; RMSE 14.98 / 17.46.
8. Show graphs `02`–`04` and be honest: residuals are not perfectly Gaussian, so the MLE
   interpretation is approximate.

## Step 6: Logistic regression (3 min)
1. Graph `12`: linear regression gives 326 negative "probabilities", hence logistic.
2. Bernoulli likelihood, no closed form, IRLS.
3. Graph `11`: the full model's MLE **does not exist** because of separation (276 one-class
   words, LP check), so ‖β‖ runs away.
4. Two fixes: the small model (a true MLE, IRLS = sklearn to 1e-15) and L2 (MAP, not MLE).
5. Graph `09` (confusion matrices) and the baseline: L2 accuracy 0.9870 vs 0.8949 for
   "always no"; F1 0.9348.

## Step 6b: Extended models (2 min), from `07_extended_models.md`
1. Graph `13`: on the score, ridge and Bayesian = OLS (nothing to shrink with 6 features),
   Huber is slightly better, and kernel ridge is best (test R² 0.813) because the rubric needs
   interactions (the kernel trick).
2. Graph `16`: Bayesian regression adds 95% intervals that cover 93.3% of test scores.
3. Graph `17`: discriminative (logistic, linear SVM, F1 0.935) beats generative (naive Bayes
   0.910, LDA 0.519).
4. Graph `18`: Bayesian logistic with the Laplace approximation fixes the missing MLE with a
   prior, but its averaged probabilities were less sharp (log loss 0.058 → 0.107).
5. SVM: max margin still works on separable data; 342 support vectors.

## Step 7: Limitations (1 min)
AI-judged labels (no human validation), weak category labels, one snapshot in time,
constructed requirements, a small regression test set (6 requirements).

## Step 8: Close (30 s)
"Phase 1 shows MLE end to end: closed form for linear regression, numerical IRLS for
logistic, and what happens when the MLE does not exist."

---

## Why notebooks *and* scripts?

| | Scripts (`src/*.py`) | Notebooks (`notebooks/*.ipynb`) |
|---|---|---|
| Purpose | Reproducible source of truth: run top to bottom, write outputs | Teaching: theory (LaTeX) next to live results and plots |
| Who reads it | The pipeline, the grader re-running it | The examiner, you in the viva |
| Logic | All maths lives here | **Imports the same functions**, so no duplicated or diverging code |

* A notebook lets you show **derivation → code → number → plot** on one screen, which is
  exactly what a viva asks for.
* Notebooks alone are fragile (hidden state, out-of-order cells), so the real computations
  live in scripts, and the notebooks are re-executed from scratch (`nbconvert --execute`)
  so every output is current.
* `02_logistic_mle.ipynb` derives the sigmoid, likelihood, gradient, Hessian, IRLS and
  separation. `03_linear_mle.ipynb` derives the Gaussian likelihood, RSS = MLE, β̂, σ̂² vs
  s², and the closed-form log-likelihood.

**Tip for the viva:** open notebook 03 and scroll; each theory cell is followed by the cell
that proves it with the real numbers.
