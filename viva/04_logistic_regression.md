# 4. Logistic regression for classification (with the real numbers)

Target: `is_pharmacy` (1 if the vendor appears in the pharmacy file; a **weak** label).
Stratified 80/20 split, seed 42: train 3,689 vendors (389 pharmacies), test 923 (97
pharmacies, prevalence 0.1051).

## 4.1 Why not linear regression for a 0/1 target? (graph `12`)

1. **Impossible predictions.** We fitted plain linear regression (a "linear probability
   model") to is_pharmacy with the same 442 columns. On the test set its predictions ran from
   −0.311 to 1.802: **326 below 0 and 18 above 1** out of 923. A probability cannot be
   negative or above 1.
2. **Wrong noise model.** y is 0 or 1, so it is Bernoulli, not Gaussian. Its variance
   p(1 − p) changes with x, which breaks linear regression's constant-variance assumption.
3. **Wrong shape.** A probability should flatten near 0 and 1; a straight line keeps going.

Logistic regression fixes all three by passing the linear score through the sigmoid:

$$z = X\beta,\qquad p_i = \sigma(z_i) = \frac{1}{1+e^{-z_i}} \in (0,1),\qquad \log\frac{p_i}{1-p_i} = z_i$$

So the model is linear in the **log-odds**, and the output is always a valid probability.

## 4.2 Bernoulli likelihood and log-likelihood

$$L(\beta)=\prod_i p_i^{\,y_i}(1-p_i)^{1-y_i},\qquad
\ell(\beta)=\sum_i\big[y_i z_i - \ln(1+e^{z_i})\big]$$

Gradient Xᵀ(y − p), Hessian −XᵀWX with W = diag(pᵢ(1 − pᵢ)). The Hessian is negative
semi-definite, so ℓ is concave: any peak is the global peak, **if a peak exists**.

## 4.3 Why there is no closed form (contrast with linear regression)

Setting the gradient to 0 gives Xᵀ(y − σ(Xβ)) = 0. β sits **inside the non-linear sigmoid**,
so unlike XᵀXβ = Xᵀy it cannot be solved with algebra. We solve it numerically with
**Newton–Raphson = IRLS** (iteratively reweighted least squares):

$$\beta^{(t+1)} = \beta^{(t)} + (X^\top W X)^{-1}X^\top(y-p)$$

Each step is a weighted least-squares fit, which is where the name comes from.

## 4.4 Why the full model's MLE does not exist (graph `11`)

**Separation:** if some direction b puts every pharmacy on one side and every non-pharmacy on
the other, pushing β further along b always increases ℓ. ℓ creeps towards 0 but never reaches
a maximum, so **no finite β maximises it**.

Real evidence from our training data:
* **276 title tokens appear in only one class**: e.g. *medicals* appears only in pharmacies,
  and *market*, *ration*, *xerox* only in non-pharmacies.
* **A joint check by linear programming** finds a separating direction (optimum 6017 > 0;
  0 would mean "no separation").
* **IRLS never converges.** ‖β‖ runs away to 650.76 and stops only because the probabilities
  hit exactly 0 or 1 in floating point.
* **Same likelihood, different answers.** IRLS and scikit-learn both reach ℓ = −103.5582,
  but their coefficients differ by up to 2705.12. On a flat ridge, many β's give the same ℓ.
* **The rank is fine** (442 of 442), so separation is the only cause.

Log-likelihoods on train, for comparison:

| model | ℓ (train) |
|---|---|
| null (intercept only) | −1242.7964 |
| full unpenalised (not a maximum) | −103.5582 |
| small-model MLE | −857.2226 |
| L2 model | −146.0539 |

## 4.5 Two fixes, reported separately

**(1) Small model where the MLE exists.** We keep at most 10 tokens, taken in order of
**training-set** frequency, and skip any that would cause separation (skipped: *market,
ration, medicals, xerox*). The chosen tokens are *shop, store, and, sri, chennai, stores,
pharmacy, s, home, centre*. The LP check finds no separation. IRLS converges in **8
iterations**, and IRLS and sklearn `penalty=None` agree to **1.78e-15**, with identical ℓ =
−857.2226. This is a true MLE, but a weak classifier.

**(2) L2 model: penalised likelihood (MAP), not MLE.** We maximise ℓ(β) − (λ/2)‖β‖². The
penalty makes the objective strictly concave, so a unique finite answer always exists. This
equals the posterior mode with a Gaussian prior on β, so it is a **MAP** estimate, not an MLE.
λ = 0.1 (C = 10) was chosen by 5-fold cross-validation on the training set only. The
hand-written penalised IRLS reproduces sklearn to about 1e-5.

## 4.6 Confusion matrix and scores (graph `09`, `10`)

Test set, threshold 0.5 (predict pharmacy if p ≥ 0.5):

| model | TN | FP | FN | TP | accuracy | precision | recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| L2 (MAP, not MLE) | 825 | 1 | 11 | 86 | 0.9870 | 0.9885 | 0.8866 | 0.9348 | 0.9842 | 0.9465 |
| Small-model MLE | 826 | 0 | 62 | 35 | 0.9328 | 1.0000 | 0.3608 | 0.5303 | 0.7438 | 0.4695 |
| Full unpenalised (invalid MLE) | 816 | 10 | 12 | 85 | 0.9762 | 0.8947 | 0.8763 | 0.8854 | 0.9762 | 0.9094 |
| Majority baseline (always 0) | 826 | 0 | 97 | 0 | 0.8949 | 0 | 0 | 0 | 0.5 | 0.1051 |

How each number is computed (L2 model):
* **Accuracy** = (TP + TN)/N = (86 + 825)/923 = 0.9870. The share of all vendors classified
  correctly.
* **Precision** = TP/(TP + FP) = 86/87 = 0.9885. When it says "pharmacy", it is almost always
  right.
* **Recall** = TP/(TP + FN) = 86/97 = 0.8866. It finds 89% of the (weak-label) pharmacies.
* **F1** = 2PR/(P + R) = 0.9348. Balances precision and recall.
* **ROC-AUC** = 0.9842: the chance that a random pharmacy gets a higher score than a random
  non-pharmacy.
* **PR-AUC** = 0.9465 against a chance level of 0.1051 (the prevalence).

**Why accuracy alone is misleading:** only 10.5% of vendors are pharmacies, so "always say
no" already scores 0.8949 accuracy with zero recall. That is why we report precision, recall,
F1 and PR-AUC next to the baseline.

Other classifiers (Fisher LDA, naive Bayes, Bayesian logistic regression via the Laplace
approximation, SVMs) are compared in `07_extended_models.md`.

## 4.7 Weak labels and the AI-judged audit

The labels come from file membership, so some are wrong. An AI judge labelled 200 test
vendors from title and address (**AI-judged labels, not ground truth**). It agreed with the
weak labels on 93.50% of the sample (98.59% after re-weighting to the test set, Cohen's κ
0.9204). Every disagreement was a weak "pharmacy" that is not one: a chemistry academy,
hospitals, a medical college, supermarkets, and so on. Against the AI labels, the L2 model's
re-weighted precision is 0.9535, recall 0.9762 and F1 0.9647.
