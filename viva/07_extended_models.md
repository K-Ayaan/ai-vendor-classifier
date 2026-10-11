# 7. Extended Unit-II models: what was added, why, and what happened

Script: `python src/extended_models.py` → `reports/extended_models_metrics.json`.
Graphs 13–19. Same data, splits and seed as Phase 1; every hyperparameter is chosen by
cross-validation **on the training split only**. No new libraries.

Each topic was added only where it answers a question Phase 1 raised. The honest results
(including where a method did *not* help) are part of the story.

---

## Regression side (target: AI-judged expected_score, not ground truth)

| model | train R² | test R² | test RMSE | test MAE |
|---|---|---|---|---|
| OLS (Gaussian MLE, Phase 1) | 0.7983 | 0.7510 | 17.4579 | 13.9775 |
| Ridge (L2, MAP) | 0.7983 | 0.7510 | 17.4579 | 13.9775 |
| Robust (Huber) | 0.7955 | 0.7630 | 17.0318 | 13.7805 |
| Bayesian linear | 0.7983 | 0.7510 | 17.4578 | 13.9639 |
| Kernel ridge (poly deg 2) | 0.8664 | 0.8082 | 15.3205 | 11.7256 |
| Kernel ridge (RBF) | 0.8731 | 0.8127 | 15.1413 | 11.4336 |

### 7.1 Ridge regression (graph `14`)
* **Idea:** minimise RSS + λ‖β‖². Closed form β̂ = (XᵀX + λI)⁻¹Xᵀy. Equivalent to MAP with
  a Gaussian prior β ~ N(0, σ²/λ · I), so it is the bridge from MLE to Bayesian.
* **Why here:** to test whether shrinking β helps on new requirements.
* **Result:** grouped 4-fold CV (whole requirements per fold) chose λ = 0.001, the smallest on
  the grid, so ridge = OLS. Graph 14 shows the coefficients only shrink once λ ≳ 10.
* **Takeaway:** with 6 well-conditioned features (κ = 114.45) and 360 rows there is nothing to
  shrink. Ridge matters when there are many correlated features (as in the 442-column
  classifier, where L2 was essential).

### 7.2 Robust regression: Huber (graph `15`)
* **Idea:** replace the squared loss with the Huber loss: squared for small residuals,
  **linear** for |r| > ε·scale. Large errors pull the line less. (Equivalent to MLE under
  a heavier-tailed noise model than Gaussian.)
* **Why here:** Phase 1 residuals are heavy-tailed (excess kurtosis 2.220) and the worst
  misfits are mislabelled vendors.
* **Result:** ε = 1.35, scale 9.2588; **121 of 360** training pairs fall in the linear part.
  Coefficients barely move (category_match even rises slightly; within_radius and is_chain
  shrink), and test R² improves a little: 0.7510 → 0.7630.
* **Takeaway:** the outliers were not distorting the fit much, but robustness is a cheap
  safeguard when labels are noisy.

### 7.3 Bayesian linear regression (graph `16`)
* **Idea:** prior β ~ N(0, λ⁻¹I) and noise precision α. The posterior is Gaussian:
  S_N = (λI + αXᵀX)⁻¹, m_N = αS_N Xᵀy. Prediction for a new x is a **distribution**:
  mean xᵀm_N, variance 1/α + xᵀS_N x. α and λ are chosen by maximising the evidence
  (empirical Bayes).
* **Why here:** a match score should come with uncertainty ("74 ± 30").
* **Result:** α = 0.004385, so noise variance 1/α = 228.0577 (very close to s² = 228.7249);
  λ = 0.001456 is a weak prior, so the posterior mean ≈ OLS. The hand-written closed-form
  posterior matches sklearn to 2.27e-13. On test, the mean predictive sd is 15.2786 and the
  95% intervals cover **93.33%** of the actual scores (target 95%).
* **Takeaway:** same point predictions as OLS, but now every vendor gets an honest interval,
  and the intervals are close to well calibrated.

### 7.4 Kernel functions, the kernel trick, kernels in GLMs (graph `13`)
* **Idea:** a kernel k(x, x′) = φ(x)ᵀφ(x′) computes an inner product in a richer feature
  space **without building φ** (the kernel trick). Kernel ridge regression is ridge in that
  space: ŷ(x) = Σᵢ aᵢ k(x, xᵢ), with a = (K + λI)⁻¹y (the dual form of a GLM with a Gaussian
  likelihood).
  * Polynomial degree 2: k = (γxᵀx′ + 1)², which adds every pairwise **interaction** (e.g.
    category_match × within_radius).
  * RBF: k = exp(−γ‖x − x′‖²), a smooth similarity.
* **Why here:** the rubric is **multiplicative** (100 only if category *and* radius *and*
  keyword match), but OLS is additive.
* **Result:** test R² 0.7510 → **0.8082** (poly) and **0.8127** (RBF); test RMSE 17.46 → 15.14.
  The train/test gap grows (RBF train R² 0.8731), so watch for overfitting.
* **Takeaway:** the biggest real gain; interactions matter for matching.

---

## Classification side (target: weak is_pharmacy labels, test set)

| model | F1 | ROC-AUC | PR-AUC | accuracy |
|---|---|---|---|---|
| Logistic L2 (MAP), Phase 1 | 0.9348 | 0.9842 | 0.9465 | 0.9870 |
| Logistic small-model MLE, Phase 1 | 0.5303 | 0.7438 | 0.4695 | 0.9328 |
| Fisher LDA (discriminant) | 0.5185 | 0.8786 | 0.6055 | 0.9296 |
| Bernoulli naive Bayes (generative) | 0.9101 | 0.9785 | 0.9365 | 0.9816 |
| Bayesian logistic (Laplace) | 0.9348 | 0.9539 | 0.9196 | 0.9870 |
| SVM linear | 0.9348 | 0.9805 | 0.9407 | 0.9870 |
| SVM RBF kernel | 0.8936 | 0.9707 | 0.9183 | 0.9783 |

### 7.5 Discriminant function: Fisher LDA
* **Idea:** a linear discriminant y(x) = wᵀx + w₀; Fisher chooses w to maximise
  between-class separation over within-class spread. Equivalently: Gaussian classes with a
  shared covariance give a linear decision boundary.
* **Result:** F1 0.5185 (TN 823, FP 3, FN 62, TP 35), the weakest.
* **Takeaway:** the Gaussian assumption is a poor fit for sparse 0/1 word indicators. A good
  example of a model whose assumptions do not match the data.

### 7.6 Probabilistic generative model: Bernoulli naive Bayes
* **Idea:** model p(x | class) and p(class), then use Bayes' rule. Naive Bayes assumes each
  word appears independently given the class:
  p(x|C) = Πⱼ θⱼᶜ^xⱼ(1 − θⱼᶜ)^(1−xⱼ), with Laplace smoothing α = 1.
* **Generative vs discriminative:** logistic regression models p(class | x) directly.
  NB models how the data are *generated*, which needs fewer data but makes a stronger
  (independence) assumption.
* **Result:** F1 0.9101 vs logistic 0.9348. It is close, but the discriminative model wins
  with 3,689 training vendors, as theory predicts for larger samples.

### 7.7 Laplace approximation + Bayesian logistic regression (graph `18`)
* **Idea:** prior β ~ N(0, λ⁻¹I). The posterior has no closed form, so approximate it by a
  Gaussian at its peak: **N(β_MAP, S)** with S = (XᵀWX + λP)⁻¹, which is the inverse
  Hessian that IRLS already computes. The predictive probability averages over β using the
  probit approximation: p = σ(μ / √(1 + πs²/8)), μ = xᵀβ_MAP, s² = xᵀSx.
* **Why here:** the prior fixes Phase 1's "MLE does not exist" problem *by design*, and the
  posterior gives uncertainty.
* **Result:** λ = 0.1 (the same prior strength as the L2 model), so the decisions are
  identical (F1 0.9348). But the predictive probabilities are pulled towards 0.5 (mean shift
  0.0545, max 0.2076), and test log loss **worsens** from 0.0583 (plug-in) to 0.1065
  (predictive); ROC-AUC drops 0.9842 → 0.9539.
* **Takeaway:** with 442 parameters and separated directions, the posterior is very wide, so
  averaging over it makes the model under-confident here. Bayesian is not automatically
  better: the prior matters.

### 7.8 Support vector machines
* **Idea:** find the boundary with the **maximum margin**; only the points on or inside the
  margin (support vectors) define it. The soft margin C trades margin width against errors.
  Kernels (RBF) make the boundary non-linear via the kernel trick.
* **Why here:** Phase 1's data are **separable**, so the logistic MLE does not exist, but
  the max-margin problem still has a unique solution.
* **Result:** linear SVM (C = 1): F1 0.9348, the same confusion matrix as L2 logistic,
  using **342 support vectors** out of 3,689. RBF SVM (C = 10): F1 0.8936 with 1,155
  support vectors. The non-linear kernel does not help with word indicators.
* **Takeaway:** a sparse, margin-based classifier matches regularised logistic regression.

---

## One-paragraph summary for the viva
"Beyond MLE, we tested the other Unit-II models where each answered a real question. For the
match score, ridge and Bayesian regression gave the same point estimates as OLS (6
well-conditioned features), but Bayesian regression added calibrated 95% intervals (93.3%
coverage). Huber was a small robustness gain, and kernel ridge was the largest gain (test R²
0.751 → 0.813) because the rubric rewards feature interactions. For classification,
discriminative models (logistic L2, linear SVM) beat generative ones (naive Bayes 0.910, LDA
0.519). The Laplace-approximated Bayesian logistic regression fixes the non-existent MLE with
a prior, but its averaged probabilities were less sharp on this data."
