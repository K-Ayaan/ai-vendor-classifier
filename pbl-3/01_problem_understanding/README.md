# 1. Problem Understanding & Dataset Familiarization (4 marks)

> What the examiner checks: can you state what Unit V (combining models) is used **for** in this
> project, why a single model is not enough for this data, and what the data can and cannot
> support?

---

## 1.1 The problem in one paragraph

PBL-1 built the two statistical parts of a vendor-matching agent, each with **one** model:

1. a **match-score regression**: predict the 0–100 score of a (buyer requirement, vendor) pair
   so vendors can be ranked (linear regression by Gaussian MLE, test R² 0.751; the best Unit II
   model, kernel ridge with an RBF kernel, reached 0.813);
2. a **pharmacy classifier**: decide whether a vendor really is a pharmacy from its name and
   pincode (L2 logistic regression, test F1 0.935).

PBL-3 asks: **can we do better by combining models instead of choosing one?** Unit V gives several
ways to combine, and they answer different questions:

| Way to combine | Unit V topic | What it combines | Question it answers here |
|---|---|---|---|
| Split the input space into regions, one simple model per region | **CART** (tree-based models) | constant predictions in leaves | Is the score a set of rules ("right category *and* inside the radius → high")? |
| Average many models trained on resampled data | **Ensemble learning**: committees, bagging, random forests | many deep trees | Can averaging remove the instability of a single tree? |
| Add basis functions one at a time, each fixing the last one's errors | **Boosting**: AdaBoost, gradient boosting; **adaptive basis-function models** | stumps / small trees | Can many weak rules (one word each) make a strong pharmacy classifier? |
| Sum of one smooth function per input | **Generalized additive models** | one curve per feature | Is the distance effect a curve rather than a line? How far does an additive model get without interactions? |
| Weight whole models by how probable they are | **Bayesian model averaging** | all 64 linear models | Which inputs does the data really need, and how sure are we? |
| Let each data point be explained by one of several models | **Conditional mixtures / mixture of experts** (Bishop 14.5) | linear experts with a gate | Are there separate regimes ("wrong category" vs "right category") with their own lines? |
| Learn the weights of several finished models | **Stacking** (ensemble learning) | OLS, kernel ridge, GAM, trees, forests, boosting | Does a learned blend beat the best single model? |

## 1.2 Why this data calls for combined models (evidence from the EDA)

These are training-split facts from [02_eda_insights](../02_eda_insights/README.md):

| EDA finding | Why a single linear model struggles | Unit V answer |
|---|---|---|
| **E01:** being inside the radius adds **+31.1** points for the right category but only **+3.0** for the wrong one | a linear model has one radius coefficient for both, so it under-predicts one cell and over-predicts the other (predicts **−6.9** for a far, wrong-category vendor) | trees split on `category_match` first, then on radius; boosting with depth ≥ 2 learns interactions; GAM needs an explicit interaction term |
| **E02:** the score is a **step** at the radius (79.2 inside, 56.9 at 1–2× the radius), not a straight line in km | a straight line in distance cannot drop at the buyer's own radius | trees and splines model steps and curves |
| **E03:** out-of-fold OLS residuals average **+10.6**, **−8.1**, **−5.4**, **+6.7** across the four cells | structured residuals mean the model is mis-specified, not just noisy | models that adapt their basis functions to the data |
| **E04:** errors of different model families correlate **0.70–0.99** (mean 0.82) | – | room for combining is real but limited: averaging helps only where errors differ |
| **E05:** a full-depth tree's prediction for the same pair moves by **7.1** points (s.d.) across bootstrap refits, against 1.9 for OLS | one tree is a high-variance estimator | bagging and random forests average that variance away |
| **E06–E07:** the best single word ("pharmacy") gets 93.5% accuracy against 89.5% for "always no"; the 17 most precise words together still miss **12.6%** of pharmacies | each word is a *weak learner*; no single rule covers every pharmacy | boosting combines hundreds of one-word stumps |
| **E08:** 41.9% of scores sit exactly on a rubric anchor, 25.8% are exactly 0, and the spread differs **2.5-fold** between cells | one Gaussian noise level is wrong | mixtures of regressions model several regimes; trees fit cell means |

## 1.3 The two datasets, unchanged from PBL-1

Both tasks reuse the PBL-1 data **and its exact splits**, so every number here is comparable with
PBL-1. See [PBL-1 §1.4](../../pbl-1/01_problem_understanding/README.md) for the field-by-field
audit of the raw Google Maps snapshot (23 JSON files, 2026-09-27, 5,439 records → 4,612 vendors).

| | Match score (regression) | Pharmacy (classification) |
|---|---|---|
| Unit | one requirement–vendor pair | one vendor |
| Rows | 450 pairs (30 requirements × 15 candidates) | 4,612 vendors |
| Inputs | 6: `distance_km`, `within_radius`, `category_match`, `keyword_in_title`, `n_categories`, `is_chain` | 441 binary columns: 421 title words (min doc. freq. 5 on train) + 20 pincodes |
| Target | `expected_score` 0–100, AI-judged against `data/annotation/RUBRIC.md` | `is_pharmacy`, the weak label from file membership (10.5% positive) |
| Split | by requirement: 24 train / 6 test requirements = 360 / 90 pairs | stratified 80/20: 3,689 / 923 |
| Hyperparameters | grouped 4-fold CV on train (whole requirements per fold) | stratified 5-fold CV on train, or out-of-bag error |
| Measures | R², RMSE, MAE; 95% CI by resampling the 6 test requirements | F1, precision, recall, ROC-AUC, PR-AUC, log loss; bootstrap CI; exact McNemar test |

## 1.4 What each Unit V method needs from the data

| Method | Needs | Available here? |
|---|---|---|
| CART, bagging, random forest | any feature types; enough rows per leaf | yes. Binary features make the split search a matrix product (§3.1) |
| AdaBoost | a weak learner better than chance on **weighted** data | yes. 20 of 441 single-word stumps beat "always no" (E06) |
| Gradient boosting | a differentiable loss | squared error (score) and logistic loss (pharmacy) |
| GAM | at least one continuous input to smooth | `distance_km` (continuous) and `n_categories` (1–10). The 4 binary inputs can only be linear terms, so the classifier (all binary) gains nothing from a GAM and none is fitted for it |
| BMA | a marginal likelihood per model and a manageable model space | 2⁶ = 64 linear models (closed-form g-prior evidence); 2¹⁰ = 1,024 logistic models on the 10-word model where the MLE exists (PBL-1 §3.2) |
| Mixture of regressions / experts | regimes with different lines | the four category × radius cells (E01, E08) |
| Stacking | out-of-fold predictions that do not leak | grouped CV for the score (a buyer's 15 pairs never split across folds) |

## 1.5 What the data does NOT contain (and what we did about it)

| Missing | Consequence | Our answer |
|---|---|---|
| Human match scores | the regression target is the AI judge's rubric scores | stated everywhere. A high R² means "learned the rubric", not "perfect matcher". The blind re-score (r = 0.976, PBL-1) bounds how well any model can do |
| True pharmacy labels | errors may be label noise, not model errors | the 200-vendor AI label audit from PBL-1 is used to check which test "errors" are really mislabelled vendors (C06) |
| Many test buyers | only **6** test requirements, so test R² is noisy | 95% confidence intervals by resampling whole requirements; paired comparisons on the same pairs |
| Repeated scores of the same pair | noise cannot be separated from bias | the bias–variance analysis reports "bias² + noise" together (§3.3) |
| Large data | 360 training pairs: flexible models can overfit | every hyperparameter by grouped CV; trees pruned by cost-complexity; boosting rounds by CV |

## 1.6 Splits and leakage control

- **Same splits as PBL-1**, recomputed by the same code (`lr.split_by_requirement`,
  `L.split`), so no test row was ever used to choose anything.
- **Grouped CV** for every score-model hyperparameter (tree α, forest m and leaf size, boosting
  depth/ν/M, GAM λ, stacking's out-of-fold predictions). A random K-fold would put pairs of the
  same buyer on both sides and pick models that are too flexible.
- **Out-of-bag error is optimistic for the score model** (the other 14 pairs of a buyer are
  usually in the bag), so it is reported but not used for selection there. For the classifier
  (independent vendors) it is valid and used for the random-forest feature-subset size.
- **The vocabulary and pincode list** are fitted on the training vendors only (PBL-1
  `FeatureBuilder`).
- **Seed 42** everywhere; re-running the scripts reproduces every number.

## 1.7 Assumptions (know these)

- **Rows are independent given the features**, except for pairs that share a requirement
  (handled by grouping).
- **Squared error** is the regression loss (as in PBL-1), so the boosting and tree results are
  comparable with OLS. **Threshold 0.5** for the classifiers, as in PBL-1.
- **Uniform prior over models** in BMA; Zellner **g-prior with g = n** (unit information) for the
  coefficients.
- **Residual s.d. floor of 1 score point** in the mixture models (scores are integers; without
  the floor a component collapses onto the 25.8% exact zeros and the likelihood goes to ∞).

## 1.8 Glossary

| Term | One-line meaning |
|---|---|
| Weak learner | a model only slightly better than chance (a one-word stump) |
| Decision stump | a tree with one split: "if word present → +1 else −1" |
| Committee / ensemble | several models whose predictions are combined |
| Bootstrap sample | n rows drawn with replacement; about 63.2% of the distinct rows appear |
| Out-of-bag (OOB) | rows not in a tree's bootstrap sample, giving a free validation prediction |
| Bagging | **b**ootstrap **agg**regat**ing**: average models trained on bootstrap samples |
| Random forest | bagging plus a random subset of m features at every split |
| Boosting | fit models in sequence, each one focusing on what the previous ones got wrong |
| Adaptive basis function | a basis function whose shape is learned from the data (a tree), not fixed in advance (a polynomial) |
| Shrinkage ν | the learning rate: each new tree is added as ν · tree |
| Cost-complexity pruning | cut back a big tree by trading training error against the number of leaves (α · leaves) |
| Additive model | y = α + Σ fⱼ(xⱼ): one function per input, no interactions |
| Backfitting | fit an additive model by smoothing each fⱼ against the partial residual in turn |
| Marginal likelihood / evidence | p(D \| M): how probable the data is under a model, averaged over its parameters |
| Posterior inclusion probability | the total posterior weight of the models that contain an input |
| Stacking | train a second-level model on the out-of-fold predictions of the first-level models |

---

## Viva questions on this section

**Q1. What is the difference between "combining models" and "choosing the best model"?**
Choosing keeps one model and discards the rest. Combining keeps several and merges their
predictions: by averaging (bagging), by weighted voting (boosting), by posterior weights (BMA) or
by learned weights (stacking). Combining helps when the models make **different** errors (E04).

**Q2. Why should trees help on this data at all? The features are only six.**
Because the effect of one feature depends on another (E01: radius matters 10× more for the right
category) and the score jumps at the radius (E02). A tree models exactly that: "if category
matches, then if inside the radius…". A linear model needs a hand-made interaction term.

**Q3. Why is BMA in the same unit as boosting if it is not really a combination?**
Bishop 14.1 makes this point: BMA assumes **one** model generated all the data and averages
because we are unsure which. With more data its weights concentrate on one model. A committee
or mixture combines models because different models are better for different inputs. I show
both behaviours on the same data (§3.6, §3.7).

**Q4. Why not use a GAM for the pharmacy classifier?**
All 441 inputs are 0/1 indicators. A smooth function of a 0/1 input is just two values, a
linear term. A logistic GAM would be identical to the PBL-1 logistic regression.

**Q5. You have 6 test requirements. Can you trust a test R² difference of 0.02?**
Not on its own. That is why every comparison has a 95% interval from resampling whole test
requirements and a paired difference on the same pairs (C01, C02). Differences smaller than
about the interval width are reported as "no clear difference".

**Q6. Why is the out-of-bag error not used to tune the score forests?**
A requirement's 15 pairs share one buyer. When a pair is out of bag, its 14 siblings are usually
in the bag, so OOB is optimistic, like a random split. Grouped CV keeps whole buyers out.
