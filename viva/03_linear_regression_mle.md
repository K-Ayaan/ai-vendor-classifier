# 3. Linear regression as maximum likelihood (with the real numbers)

Target y = **AI-judged expected_score** (0–100) for 450 requirement–vendor pairs.
Train = 360 pairs (24 requirements), test = 90 pairs (6 requirements). p = 7 columns.

## 3.1 The model

$$y = X\beta + \varepsilon,\qquad \varepsilon \sim N(0,\ \sigma^2 I)$$

Each score is "a straight-line combination of the features + Gaussian noise". Two unknowns:
the coefficients β (7 numbers) and the noise variance σ² (1 number).

## 3.2 The likelihood and log-likelihood

Each yᵢ is Normal with mean xᵢᵀβ and variance σ², and the rows are independent, so

$$L(\beta,\sigma^2)=\prod_{i=1}^{n}\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\Big(-\frac{(y_i-x_i^\top\beta)^2}{2\sigma^2}\Big)$$

Taking logs turns the product into a sum:

$$\ell(\beta,\sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) \;-\; \frac{1}{2\sigma^2}\underbrace{\sum_i (y_i - x_i^\top\beta)^2}_{\text{RSS}(\beta)}$$

We use the log because it is easier to differentiate and avoids multiplying 360 tiny
numbers together, and log is increasing, so the maximiser is the same.

## 3.3 Why minimising RSS = maximising the likelihood (say this in the viva)

1. Look at ℓ(β, σ²). β appears **only** in the second term, −RSS(β)/(2σ²).
2. The first term −(n/2)ln(2πσ²) does not depend on β, so it cannot change which β is best.
3. 1/(2σ²) is a **positive** constant (σ² > 0). Maximising "−(positive constant) × RSS" is
   the same as **minimising RSS**.
4. So for any σ², arg max_β ℓ = arg min_β RSS(β). **The Gaussian MLE of β is exactly the
   least-squares estimate.**

Caveat: this equivalence needs the **Gaussian** error assumption. With other noise
distributions the MLE is a different estimator (e.g. Laplace noise gives least absolute
deviations).

## 3.4 The closed form for β̂

Set the gradient of RSS to zero: −2Xᵀ(y − Xβ) = 0 ⇒ **XᵀXβ = Xᵀy** (normal equations), so

$$\hat\beta = (X^\top X)^{-1}X^\top y$$

In code we solve it with `numpy.linalg.lstsq` (QR/SVD) instead of inverting XᵀX, because
inverting squares the condition number: κ(XᵀX) = κ(X)². Here **κ(X_train) = 114.45**, so it is
well-conditioned, and all three methods agree: |β̂_inverse − β̂_lstsq| ≤ 7.46e-14,
|β̂_sklearn − β̂_lstsq| ≤ 2.13e-14.

## 3.5 The real coefficients (graph `05_coefficients`)

| feature | β̂ | SE | t = β̂/SE | meaning |
|---|---|---|---|---|
| intercept β₀ | 0.2962 | 2.3888 | 0.12 | predicted score when every feature is 0 |
| distance_km | −0.1568 | 0.0758 | −2.07 | each extra km lowers the score by about 0.16 |
| within_radius | 18.1357 | 2.2583 | 8.03 | being inside the buyer's limit adds about 18 points |
| category_match | 59.4411 | 1.6868 | 35.24 | the right kind of shop adds about 59 points (largest effect) |
| keyword_in_title | 15.4676 | 4.9057 | 3.15 | matching the preference adds about 15 points |
| n_categories | −3.2396 | 0.6550 | −4.95 | each extra listed category lowers the score by about 3 |
| is_chain | 3.4746 | 4.0048 | 0.87 | small and not distinguishable from 0 |

SE = √diag(s²(XᵀX)⁻¹); valid only if the Gaussian, constant-variance assumptions hold.

**About β₀ = 0.2962.** It is the prediction when *all* features are 0, but no real vendor has
`n_categories = 0` (the minimum is 1), so β₀ alone is not a real vendor. It is an anchor
that lets the other coefficients act as "adjustments".

**Worked predictions** (plugging into ŷ = β₀ + Σβⱼxⱼ, with n_categories = 1, not a chain):

| vendor | ŷ | rubric anchor |
|---|---|---|
| right category, 1 km, inside radius, keyword match | 89.9441 | 100 |
| right category, 1 km, inside radius, no keyword | 74.4765 | 75 |
| right category, 10 km, outside radius | 54.9300 | 50 |
| wrong category, 1 km, inside radius | 15.0354 | 0–25 |
| wrong category, 30 km, outside radius | −7.6465 | 0 (impossible negative) |

The model rebuilds the rubric anchors, which is **expected**: the AI judge used the same
criteria. That is why high R² here is not proof of a good matcher.

## 3.6 Variance: σ̂²_MLE vs s²

Differentiate ℓ with respect to σ² and set it to zero: −n/(2σ²) + RSS/(2σ⁴) = 0 ⇒

$$\hat\sigma^2_{MLE} = \frac{RSS}{n} = \frac{80739.8798}{360} = 224.2774,\qquad \hat\sigma_{MLE}=14.9759$$

**Theory:** this is biased: E[σ̂²_MLE] = ((n−p)/n)σ², because fitting 7 coefficients
"uses up" 7 degrees of freedom. The unbiased version divides by n − p:

$$s^2 = \frac{RSS}{n-p} = \frac{80739.8798}{353} = 228.7249,\qquad s = 15.1237$$

**Reality check:** σ̂²_MLE / s² = 0.980556 = (n−p)/n = 353/360, exactly as theory says.

**MSE vs σ̂²_MLE:** train MSE = RSS/n = 224.2774 = σ̂²_MLE. Same number, different meaning:
MSE is a *descriptive loss* (how wrong were the predictions; can be computed on any data,
e.g. test MSE = 304.7785), while σ̂²_MLE is an *estimate of a model parameter* (the noise
variance), it is biased, and it only means something under the Gaussian model.

## 3.7 The maximised log-likelihood, with the numbers plugged in

At β̂ and σ̂²_MLE:

$$\ell = -\frac{n}{2}\ln(2\pi\hat\sigma^2) - \frac{RSS}{2\hat\sigma^2} = -1305.1370 + (-180.0000) = -1485.1370$$

The second term is **always −n/2** at the MLE, because RSS/(2·RSS/n) = n/2 = 180. That gives
the closed form

$$\ell_{max} = -\frac{n}{2}\big(\ln(2\pi\hat\sigma^2) + 1\big) = -1485.1370$$

and the direct sum Σ log φ(eᵢ; 0, σ̂) also gives −1485.1370. Graph `06_log_likelihood_sigma2`
shows ℓ as a function of σ²: it peaks exactly at σ̂²_MLE = 224.28, and s² = 228.72 sits just
to the right (not the peak).

## 3.8 Error metrics and R²

| quantity | train | test |
|---|---|---|
| RSS = Σ(y − ŷ)² | 80739.8798 | 27430.0652 |
| MSE = RSS/n | 224.2774 | 304.7785 |
| RMSE = √MSE (score points) | 14.9759 | 17.4579 |
| MAE = mean abs error | 11.3600 | 13.9775 |
| R² = 1 − RSS/TSS | **0.7983** | **0.7510** |

R²_train = 1 − 80739.8798 / 400293.6000 = 0.7983, where TSS = Σ(y − ȳ)² is the total
variation. **The model explains about 80% of the variation in the AI judge's scores on
training pairs and 75% on unseen requirements.** The test error is a little higher, which is
normal; the gap is small, so there is no serious overfitting (graph `07`).

## 3.9 Theory vs reality (what the diagnostics show)

| Assumption (theory) | What we see (reality) | Graph |
|---|---|---|
| Errors are Gaussian | Heavier tails: excess kurtosis 2.220, Shapiro-Wilk p = 3.59e-08 (normality rejected) | 03, 04 |
| Constant variance | Residuals form bands (the target clusters at rubric anchors 0/25/50/75/100) | 02 |
| Mean is linear and unbounded | Scores are bounded 0–100 but 61 train / 14 test predictions are below 0 | 01, 02 |
| Rows independent | The 15 pairs of one requirement share a buyer, so the split holds out whole requirements | |
| σ̂²_MLE biased by (n−p)/n | Confirmed exactly: 0.980556 | 06 |

**Conclusion:** least squares is still the best straight-line fit (minimum RSS), but the
"least squares = MLE" interpretation and the SEs are only approximately valid here.

**Extensions** (ridge, robust Huber, Bayesian linear regression, kernel ridge) are in
`07_extended_models.md`.

**The biggest misfits** (largest negative residuals) are vendors whose source-file category
"matches" but which the judge recognised as the wrong shop from the title (e.g. a handicraft
shop listed under fruits). So the AI judge did not just copy the features.
