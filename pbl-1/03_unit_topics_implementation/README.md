# 3. Assigned Unit Topic Implementation: Regression and Classification (4 marks)

> What the examiner checks: you implemented the unit's methods **yourself** (not only called a
> library), you can derive them, and you can connect each formula to code and to a number.

**Where the code lives**

| Topic | File · function |
|---|---|
| Shared features | `src/matching/features.py` · `pair_features`, `design_matrix`, `FeatureBuilder` |
| Linear regression (Gaussian MLE / least squares) | `src/linear_regression_mle.py` · `fit_ols`, `gaussian_mle`, `fit_metrics`, `split_by_requirement` |
| Diagnostics plots | `src/linear_regression_plots.py`, graphs `01`–`08` |
| Logistic regression (MLE, IRLS) | `src/logistic_regression_classification.py` · `irls`, `log_likelihood`, `detect_separation`, `separation_lp`, `fit_small_mle`, `choose_c` |
| Ridge, Huber, Bayesian linear, kernel ridge, LDA, naive Bayes, Laplace, SVM | `src/extended_models.py` · `run_regression`, `bayesian_posterior`, `run_classification`, `laplace_bayesian_logistic` |
| Derivations with live numbers | `notebooks/03_linear_mle.ipynb`, `notebooks/02_logistic_mle.ipynb` |

---

## 3.0 Unit-I foundations used throughout

If the examiner asks about the basics, every one of these is in the project:

| Concept | Where it appears |
|---|---|
| Data cleaning & preprocessing | `preprocessing.py`: merge duplicates, Unicode title cleaning, pincode extraction and imputation |
| Feature engineering | haversine distance, radius flag, category match, keyword flag, word indicators, pincode one-hot |
| Train/test split | stratified 80/20 (classification); grouped by requirement 24/6 (regression) |
| Cross-validation | 5-fold stratified CV for C (logistic); grouped 4-fold CV for ridge/kernel hyperparameters |
| Bias–variance / overfitting | train vs test gaps (R² 0.7983 → 0.7510); kernel models have bigger gaps; L2 shrinks variance |
| Evaluation metrics | R², RMSE, MAE; accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss |
| Baselines | predict-the-mean (regression), always-no and a keyword rule (classification) |
| Leakage control | every vocabulary/encoder/CV fitted on train only; group split by buyer |

---

## 3.1 Linear regression as Gaussian maximum likelihood

**Model.** y = Xβ + ε, ε ~ N(0, σ²I). X is 360 × 7 (intercept + distance_km, within_radius,
category_match, keyword_in_title, n_categories, is_chain).

**Log-likelihood.**
ℓ(β, σ²) = −(n/2) ln(2πσ²) − RSS(β)/(2σ²), with RSS(β) = ‖y − Xβ‖².

**Why least squares = MLE.** β appears only in −RSS/(2σ²), and 1/(2σ²) > 0, so maximising ℓ over
β is exactly minimising RSS. (This needs Gaussian noise: Laplace noise would give least
absolute deviations instead.)

**Solving.** ∂RSS/∂β = −2Xᵀ(y − Xβ) = 0 ⇒ normal equations XᵀXβ = Xᵀy ⇒ β̂ = (XᵀX)⁻¹Xᵀy.
In code we use `np.linalg.lstsq` (QR/SVD), because forming (XᵀX)⁻¹ squares the condition
number. κ(X) = 114.45 (well-conditioned), and three solvers agree: inverse vs lstsq 7.46e-14,
sklearn vs lstsq 2.13e-14.

**Noise variance.** ∂ℓ/∂σ² = 0 ⇒ σ̂²_MLE = RSS/n = 80739.88/360 = **224.2774**. It is biased:
E[σ̂²] = (n−p)/n · σ². The unbiased s² = RSS/(n−p) = 80739.88/353 = **228.7249**. Ratio
0.980556 = 353/360 exactly.

**Maximised log-likelihood.** ℓ_max = −(n/2)(ln 2πσ̂² + 1) = **−1485.1370**. The second term
is always −n/2 = −180 at the MLE. Checked two ways: closed form = direct Σ log φ.

**Fitted coefficients** (graph `05`; SE = √diag(s²(XᵀX)⁻¹)):

| term | β̂ | SE | t |
|---|---|---|---|
| intercept | 0.2962 | 2.3888 | 0.12 |
| distance_km | −0.1568 | 0.0758 | −2.07 |
| within_radius | 18.1357 | 2.2583 | 8.03 |
| category_match | **59.4411** | 1.6868 | **35.24** |
| keyword_in_title | 15.4676 | 4.9057 | 3.15 |
| n_categories | −3.2396 | 0.6550 | −4.95 |
| is_chain | 3.4746 | 4.0048 | 0.87 |

**Results.** R² train **0.7983**, test **0.7510**; RMSE 14.98 / 17.46; MAE 11.36 / 13.98.

**Diagnostics** (graphs `02`–`04`): residual skew −0.311, excess kurtosis 2.220, Shapiro-Wilk
p = 3.59e-08 (normality rejected); 61 train / 14 test predictions below 0. So least squares is
still the best linear fit, but the Gaussian-MLE interpretation and the SEs are approximate.

---

## 3.2 Logistic regression for classification, fitted by MLE

**Why not linear regression for 0/1?** We tried it (graph `12`): on the test set its
"probabilities" ranged from −0.311 to 1.802 (326 below 0, 18 above 1). A 0/1 target is
Bernoulli, not Gaussian.

**Model.** p = σ(Xβ) = 1/(1 + e^(−Xβ)); log-odds are linear: log(p/(1−p)) = Xβ.

**Likelihood.** L(β) = Πpᵢ^yᵢ(1−pᵢ)^(1−yᵢ) ⇒ ℓ(β) = Σ[yᵢzᵢ − ln(1 + e^zᵢ)], z = Xβ.
Gradient Xᵀ(y − p); Hessian −XᵀWX, W = diag(pᵢ(1−pᵢ)) ⇒ ℓ is **concave**, so any maximum is
global, **if it exists**.

**No closed form** (β is inside the sigmoid), so we wrote **Newton–Raphson = IRLS** by hand:
β ← β + (XᵀWX)⁻¹Xᵀ(y − p), with step-halving so each step increases ℓ. Each step is a weighted
least-squares solve, hence "iteratively reweighted least squares".

**Design matrix.** 421 title words (min_df 5 on train) + 20 pincodes + intercept; 6 word columns
that were exact linear combinations of others were dropped by a Gram–Schmidt pass, so X_train
is **full rank (442/442)**.

**Discovery: the MLE does not exist** (graph `11`):
- 276 single-feature separators (words seen in only one class, as E07 predicted);
- a **linear program** finds a jointly separating direction (optimum 6017 > 0);
- IRLS never converges: ‖β‖ grows to 650.76 and stops only through floating-point saturation;
- IRLS and sklearn reach the same ℓ = −103.5582 with coefficients up to **2705.12** apart: a flat
  ridge with no unique peak.

**Fix 1: a small model where the MLE exists.** Take words in training-frequency order and keep
one only if the design stays full rank and **not separated** (LP check). Skipped: market,
ration, medicals, xerox. Kept 10: shop, store, and, sri, chennai, stores, pharmacy, s, home,
centre. IRLS converges in **8 iterations** and equals sklearn `penalty=None` to **1.78e-15**,
ℓ = −857.2226. A true MLE, but weak (F1 0.5303).

**Fix 2: L2-penalised likelihood = MAP.** Maximise ℓ(β) − (λ/2)‖β‖² (intercept unpenalised).
This is strictly concave, so a unique finite solution always exists; it equals the posterior
mode under a Gaussian prior, so it is **MAP, not MLE**. λ = 0.1 (C = 10) chosen from 13 values
by 5-fold stratified CV log-loss **on train**. Hand-written penalised IRLS matches sklearn to
1.37e-5.

**Results (test, threshold 0.5):** L2 TN 825, FP 1, FN 11, TP 86 → accuracy 0.9870, precision
0.9885, recall 0.8866, **F1 0.9348**, ROC-AUC 0.9842, PR-AUC 0.9465. Majority baseline accuracy
0.8949 (F1 0).

---

## 3.3 Further Unit-II models (same data, splits and seed)

| Topic | Key formula / idea | Setting | Result |
|---|---|---|---|
| **Ridge regression** | min RSS + λ‖β‖²; β̂ = (XᵀX + λI)⁻¹Xᵀy; MAP with a Gaussian prior | grouped 4-fold CV → λ = 0.001 | test R² 0.7510 (= OLS: nothing to shrink with 6 features) |
| **Robust (Huber)** | squared loss for small residuals, linear beyond ε·scale | ε = 1.35, scale 9.2588 | 121/360 pairs in the linear zone; test R² **0.7630** |
| **Bayesian linear regression** | prior N(0, λ⁻¹I); posterior S_N = (λI + αXᵀX)⁻¹, m_N = αS_N Xᵀy; predictive var 1/α + xᵀS_N x | α, λ by evidence maximisation | 1/α = 228.06 ≈ s²; 95% intervals cover **93.33%** of test scores |
| **Kernel ridge (kernel trick)** | ŷ(x) = Σ aᵢk(x, xᵢ), a = (K + λI)⁻¹y | poly deg 2 (α = 3.162); RBF (α = 0.1, γ = 0.0316) | test R² **0.8082** / **0.8127**, the best |
| **Fisher LDA** | maximise between-class / within-class spread; shared-covariance Gaussians | Ledoit-Wolf shrinkage | F1 0.5185 (Gaussian assumption fits 0/1 words poorly) |
| **Bernoulli naive Bayes** (generative) | p(x\|C) = Πθⱼ^xⱼ(1−θⱼ)^(1−xⱼ), Bayes' rule | Laplace smoothing α = 1 | F1 0.9101 |
| **Bayesian logistic (Laplace approx.)** | posterior ≈ N(β_MAP, (XᵀWX + λP)⁻¹); probit predictive σ(μ/√(1+πs²/8)) | λ = 0.1 | same decisions (F1 0.9348); log loss 0.0583 → 0.1065 (less sharp) |
| **SVM** | maximum margin; soft margin C; kernels | linear C = 1; RBF C = 10 | linear F1 0.9348 with 342 support vectors; RBF 0.8936 (1,155 SVs) |

Each model answered a question Phase 1 raised:
- Ridge: does shrinkage help? (No: 6 well-conditioned features.)
- Huber: do heavy tails hurt? (A little; +0.012 R².)
- Bayesian regression: can we give uncertainty? (Yes, well calibrated.)
- Kernels: does the E10 interaction matter? (Yes, the biggest gain.)
- LDA vs naive Bayes vs logistic: generative or discriminative? (Discriminative wins.)
- Laplace: can a prior fix the missing MLE? (Yes, but it was less sharp.)
- SVM: does max-margin work on separable data? (Yes, it matches L2.)

---

## 3.4 Engineering quality (marks for "implementation")

- **Hand-written + library cross-checks** everywhere: OLS (3 solvers), IRLS vs sklearn
  (1.78e-15), penalised IRLS vs sklearn (1.37e-5), Bayesian posterior vs sklearn (2.27e-13).
- **Reproducible:** seed 42; raw data hash-verified; reruns give byte-identical outputs.
- **One shared feature module** for training and the prototype app (`src/matching/features.py`),
  tested: the app's features equal the training features on all 450 pairs.
- **Tests:** `python -m pytest tests -q` → 36 passed.

---

## Viva questions on this section

**Q1. Derive the OLS estimator.**
RSS = (y − Xβ)ᵀ(y − Xβ). Gradient −2Xᵀ(y − Xβ) = 0 ⇒ XᵀXβ = Xᵀy ⇒ β̂ = (XᵀX)⁻¹Xᵀy, which
needs XᵀX invertible (full column rank: here rank 7 of 7).

**Q2. Why is σ̂²_MLE biased, and by how much here?**
β̂ is fitted to the same data, using up p = 7 degrees of freedom, so residuals are smaller than
the true errors. E[σ̂²] = (n−p)/n·σ² = 353/360 σ². Here 224.28 vs unbiased 228.72.

**Q3. Why can't logistic regression be solved in closed form?**
The score equations Xᵀ(y − σ(Xβ)) = 0 are non-linear in β. We solve them with Newton's method;
each Newton step is a weighted least-squares problem (IRLS).

**Q4. What is separation, and how did you detect it?**
Some direction separates the classes perfectly, so ℓ keeps increasing along it and has no
maximiser. Detected three ways: 276 one-class word columns, a linear program finding a
separating direction (optimum 6017), and IRLS coefficients running away (‖β‖ 650.76).

**Q5. Is your L2 model an MLE?**
No. It maximises a penalised likelihood, which equals the posterior mode under a Gaussian
prior: a MAP estimate. The small 10-word model is the true MLE.

**Q6. What is the kernel trick?**
Computing inner products φ(x)ᵀφ(x′) in a high-dimensional feature space via a kernel k(x, x′)
without building φ. A degree-2 polynomial kernel includes every pairwise interaction, like
category × radius from E10.

**Q7. Generative vs discriminative?**
Generative (naive Bayes, LDA) models p(x|C) and p(C), then applies Bayes' rule. Discriminative
(logistic, SVM) models p(C|x) or the boundary directly. With 3,689 training vendors the
discriminative models won (F1 0.935 vs 0.910 and 0.519).

**Q8. What does the Laplace approximation do?**
It replaces an intractable posterior with a Gaussian centred at the MAP, with covariance equal
to the inverse Hessian of the negative log-posterior. That Hessian is (XᵀWX + λP), which
IRLS already computes.

**Q9. How did you choose hyperparameters without touching the test set?**
Cross-validation on the training split only: stratified 5-fold for logistic C, grouped 4-fold
(whole requirements per fold) for ridge and kernel parameters, and evidence maximisation for
Bayesian regression.
