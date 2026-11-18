# 3. Assigned Unit Topic Implementation: Unit V, Combining Models (4 marks)

> What the examiner checks: you implemented the unit's methods **yourself** (not only called a
> library), you can derive them, and you can connect each formula to code and to a number.

**Where the code lives** (own implementations in numpy/scipy; scikit-learn only to cross-check)

| Topic | File · class / function |
|---|---|
| CART (regression + classification), pruning, importance | `src/ensemble/tree.py` · `CART.fit`, `_best_split`, `pruning_path`, `prune`, `feature_importances` |
| Committees: bagging, random forests, out-of-bag | `src/ensemble/committee.py` · `Forest` |
| Boosting: AdaBoost with stumps | `src/ensemble/boosting.py` · `fit_stump`, `AdaBoost` |
| Boosting: gradient boosting (squared, logistic) | `boosting.py` · `GradientBoosting` |
| Generalized additive model | `src/ensemble/gam.py` · `Term`, `GAM.fit` (direct), `GAM.backfit` |
| Bayesian model averaging | `src/ensemble/bma.py` · `LinearBMA` (g-prior, BIC), `LogisticBMA` |
| Conditional mixtures, mixture of experts | `src/ensemble/mixture.py` · `mix_linreg`, `mixture_of_experts` |
| Experiments | `src/pbl3_unit5.py` → [unit5_numbers.md](unit5_numbers.md), figures `U5_01`–`U5_12` |
| Correctness tests | `tests/test_ensemble.py` (10 tests, all pass) |

**Data** (identical to PBL-1): match score = 360 train / 90 test pairs, 6 inputs, split by
requirement; pharmacy = 3,689 / 923 vendors, 441 binary columns, stratified split. **Every
hyperparameter is chosen on the training split only**: grouped 4-fold CV for the score
model (whole buyers per fold), stratified 5-fold CV or out-of-bag error for the classifier.

Reference points from PBL-1 (test): OLS R² **0.751**; kernel ridge RBF R² **0.813**; L2 logistic
F1 **0.935**.

---

## 3.0 One idea behind all of Unit V: adaptive basis-function models

Linear regression uses **fixed** basis functions: y = Σⱼ wⱼ φⱼ(x), with φ chosen in advance.
Unit V **learns the basis functions from the data** (Murphy ch. 16):

  f(x) = w₀ + Σₘ wₘ φ(x; vₘ)

| Model | basis function φ(x; v) | how v (and w) are fitted |
|---|---|---|
| CART | indicator of a region (a leaf) | greedy recursive splitting |
| Random forest | an entire tree | each tree on a bootstrap sample, then averaged (w = 1/M) |
| AdaBoost | a stump ±1 | stagewise: minimise exponential error given the earlier terms |
| Gradient boosting | a small tree | stagewise: fit the negative gradient of the loss |
| GAM | spline in one variable | penalised least squares (backfitting) |
| Mixture of experts | a linear model × a gate | EM |

All the **combining** in Unit V is one of four kinds: *partition* (trees), *average* (committees,
BMA), *add sequentially* (boosting, GAM), *gate* (mixtures of experts), plus *learn the
weights* (stacking).

---

# Part A: Tree-based models

## 3.1 CART (classification and regression trees)

**Model.** Partition the input space into M axis-aligned regions R₁…R_M and predict a constant
in each: f(x) = Σₘ cₘ I(x ∈ Rₘ). For squared error the best constant is the mean of the
training targets in the region; for classification, the class proportions.

**Greedy growth.** Finding the best tree is NP-hard, so grow greedily. At a node with rows S,
try every split "xⱼ ≤ s" and take the one with the biggest drop in impurity:

  Δ = I(S) − I(S_left) − I(S_right),

with I = Σᵢ wᵢ(yᵢ − ȳ)² (regression) or I = W·(1 − Σₖ pₖ²) (Gini). Thresholds are midpoints
between consecutive distinct values.

**Two search paths in `_best_split`** (they give identical answers):
- *sort path* (any column): sort once, then cumulative sums give the left/right statistics at
  every threshold in one pass;
- *binary path* (0/1 columns): left = {x = 0}, right = {x = 1}, so every statistic is a matrix
  product Xᵀw, Xᵀ(wy), Xᵀ(wy²). On the 441-column pharmacy data a full tree takes 0.3 s.

**Cost-complexity (weakest-link) pruning.** A full tree overfits (train R² 1.000). Minimise

  R_α(T) = Σ_leaves R(t) + α · |leaves(T)|.

Collapsing internal node t into a leaf costs R(t) − R(T_t) and saves |leaves(T_t)| − 1 leaves,
so it pays off once α ≥ g(t) = (R(t) − R(T_t)) / (|leaves(T_t)| − 1). Repeatedly collapsing the
node with the smallest g(t) gives a **nested** sequence of optimal subtrees (Breiman 1984).
α is then chosen by CV.

**On the match score** (figure U5_01):

| tree | leaves | test R² | test RMSE |
|---|---|---|---|
| full depth | 258 (depth 20) | 0.7348 | 18.01 |
| pruned, α by min grouped-CV error (α = 21.51) | **4** | **0.7700** | 16.78 |
| pruned, 1-SE rule (α = 75.48) | 3 | 0.8222 | 14.75 |

- The pruning path has **144** nested subtrees. Own pruned tree = scikit-learn's with the same α
  (max |Δ| 0.0), and the path itself matches scikit-learn's `cost_complexity_pruning_path`
  (test).
- The chosen tree is: *category matches?* → *inside the radius?* → … which is E01 written as
  rules. The pruned 4-leaf tree already beats OLS (0.770 vs 0.751): the main pattern is an
  interaction, which a tree represents with 3 splits.
- The 1-SE tree is better on test, but it was not the rule chosen in advance, so 0.7700 is the
  reported number. (With 6 test buyers, a 0.05 gap is within noise; see §4.)
- Impurity importance: `category_match` 0.684, `within_radius` 0.132, `distance_km` 0.121.

**On the pharmacy data:** full tree 528 leaves (depth 171, an extreme chain because each word
peels off a few vendors), F1 0.8832. Pruned by 5-fold CV misclassification to **36 leaves**:
F1 **0.9110**. The top of the tree reads: *"pharmacy" in the name → pharmacy (unless also
"pet"); else "medicals" → pharmacy; else "medical" → pharmacy; else almost surely not.*

---

# Part B: Committees and ensemble learning

## 3.2 Bagging and random forests

**Committee.** y_COM(x) = (1/M) Σₘ yₘ(x). Bishop 14.2: write each model's error
eₘ(x) = yₘ(x) − h(x). Then

  E_AV = (1/M) Σₘ E[eₘ²],    E_COM = E[((1/M) Σₘ eₘ)²].

If the errors had zero mean and were uncorrelated, E_COM = E_AV / M. They are correlated, so in
general, for equal variances and average correlation ρ:

  Var(average) = ρσ² + (1 − ρ)σ²/M,

and always E_COM ≤ E_AV (Cauchy–Schwarz). **Bagging** builds diversity by training each tree on a
bootstrap sample. A **random forest** also draws a random subset of m features at every split,
which lowers ρ.

**Implementation.** A bootstrap sample is passed to the tree as integer weights (the count of
each row), which is how scikit-learn does it too. Rows with count 0 (≈ e⁻¹ = 36.8%) are
**out-of-bag** (OOB) and get a free validation prediction.

**On the match score** (500 trees; m and min leaf by grouped CV, figure U5_02):

| committee | test R² | E_AV (mean single-tree MSE) | E_COM | E_COM / E_AV | mean error corr ρ |
|---|---|---|---|---|---|
| bagged trees (m = 6, min leaf 5) | 0.7742 | 321.68 | 276.39 | 0.859 | **0.867** |
| random forest (m = **3** of 6, min leaf 1) | **0.7991** | 361.73 | 245.89 | **0.680** | **0.698** |

The random forest's **individual** trees are *worse* (E_AV 361.7 vs 321.7), because each split
sees only 3 features, but they are **less correlated** (0.70 vs 0.87), so the committee is
*better*. That is the random-forest argument shown in numbers. Theory check:
ρ + (1 − ρ)/500 = 0.699 against the observed E_COM / E_AV = 0.680.

- OOB R² is 0.853, much higher than test 0.799: OOB is **optimistic** here because a pair's 14
  siblings (same buyer) are usually in the bag. That is why grouped CV, not OOB, chose m.
- scikit-learn's RF with the same settings: test R² 0.7926 (different random draws).

**On the pharmacy data** (m by OOB log loss, valid here because vendors are independent):

| m (features per split) | OOB log loss | OOB F1 |
|---|---|---|
| **21 (√441)** | **0.0834** | 0.9171 |
| 44 | 0.1219 | 0.9163 |
| 132 | 0.1606 | 0.9111 |
| 441 (bagging) | 0.1759 | 0.8796 |

Test: random forest F1 **0.9247**, log loss 0.0775; bagged trees F1 0.8958. Top importances:
pharmacy 0.334, medicals 0.245, medical 0.089, medplus 0.041, apollo 0.035.

**Note on an implementation difference.** scikit-learn's RF (same m) gets F1 0.9005 and log loss
**0.1771**, against 0.0775 for mine. When the m sampled features at a node allow no split,
scikit-learn keeps drawing more features. Mine makes the node a leaf. My trees therefore keep
more impure leaves and output less extreme, better-calibrated probabilities. Both are valid
random forests.

## 3.3 Bias–variance decomposition (why committees work)

The training set is resampled 30 times (bootstrap over the 24 training **requirements**).
Each model is refitted every time, and the test predictions are decomposed (figure U5_03):

| model | variance | bias² + noise | expected test MSE |
|---|---|---|---|
| OLS | **17.29** | 315.56 | 332.86 |
| full-depth tree | **146.65** | 305.35 | 452.00 |
| pruned tree | 47.92 | 260.83 | 308.75 |
| bagged trees (50) | 68.69 | 296.57 | 365.26 |
| random forest (50) | 27.03 | **238.47** | **265.50** |
| gradient boosting | 26.75 | 238.69 | **265.44** |

- OLS: **low variance, high bias** (it cannot bend to the interaction).
- Full tree: bias about as high as OLS here, but **8.5× the variance**.
- Bagging halves the tree's variance (146.7 → 68.7). The random forest cuts it to 27.0 and
  **also** has lower bias, because the feature subsampling stops the dominant
  `category_match` from being the only first split.
- Boosting reaches the same point from the other side: it starts from a high-bias stump model
  and **reduces bias** step by step, with shrinkage keeping its variance low.
- "Noise" cannot be separated from bias here: that would need several scores of the same pair.

## 3.4 Stacking (learned combination)

Level 0: six models (OLS, kernel ridge RBF, GAM, pruned CART, random forest, gradient boosting).
Their **out-of-fold** predictions on train come from grouped 4-fold CV. Level 1: non-negative
least squares on those OOF predictions (Breiman 1996, "stacked regressions"):

  min_w ‖y − Z w‖²  s.t. w ≥ 0.

| base model | OOF RMSE | NNLS weight | unconstrained LS weight |
|---|---|---|---|
| OLS | 15.36 | 0.000 | −0.153 |
| Kernel ridge RBF | 13.06 | **0.401** | 0.433 |
| GAM | 15.08 | 0.000 | 0.139 |
| CART pruned | 14.06 | 0.010 | 0.050 |
| Random forest | 13.43 | 0.000 | **−0.325** |
| Gradient boosting | **12.86** | **0.594** | 0.874 |

Test: NNLS stack R² **0.8056**; unconstrained 0.8015; plain average of the six 0.8025.

- The stack keeps only the two **most different good** models: a smooth kernel model and a
  piecewise-constant boosted tree model. Their weights sum to 1.005 (unconstrained by design).
- Unconstrained weights are negative for OLS and RF: least squares exploits their correlation
  with the other members. That is overfitting at level 1, and the reason for w ≥ 0.

**Pharmacy stacking:** logistic regression on the base models' logits (5-fold OOF): see
[unit5_numbers.md](unit5_numbers.md) §C6. The meta-model puts most weight on gradient
boosting and L2 logistic.

---

# Part C: Boosting

## 3.5 AdaBoost (Bishop 14.3)

For t ∈ {−1, +1} and a weak learner yₘ(x) ∈ {−1, +1}:

1. wₙ = 1/N.
2. For m = 1…M:
   - fit yₘ to minimise the **weighted** error Σₙ wₙ I(yₘ(xₙ) ≠ tₙ);
   - εₘ = weighted error / Σw;  αₘ = ln((1 − εₘ)/εₘ);
   - wₙ ← wₙ exp(αₘ I(yₘ(xₙ) ≠ tₙ)) (misclassified vendors gain weight).
3. Predict sign(f(x)), f(x) = ½ Σₘ αₘ yₘ(x).

**Why it works: stagewise minimisation of the exponential error.**
E = Σₙ exp(−tₙ f(xₙ)). With the first m − 1 terms fixed, E = Σₙ wₙ⁽ᵐ⁾ exp(−½ αₘ tₙ yₘ(xₙ)),
with wₙ⁽ᵐ⁾ = exp(−tₙ f_{m−1}(xₙ)). Splitting into correct and wrong points,

  E = e^{−αₘ/2} Σₙ wₙ + (e^{αₘ/2} − e^{−αₘ/2}) Σₙ wₙ I(yₘ(xₙ) ≠ tₙ).

So for any αₘ > 0 the best yₘ is the one with the smallest **weighted error**, and
∂E/∂αₘ = 0 gives αₘ = ln((1 − εₘ)/εₘ).

With normalised weights E_M / N = Πₘ 2√(εₘ(1 − εₘ)) **exactly**. Since I(miss) ≤ exp(−t f),
this bounds the training error. `test_adaboost_exponential_loss_equals_product_bound` checks the
identity to 1e-10, that E falls every round, and that training error ≤ E/N. The minimiser of E is
f = ½ ln(p/(1−p)), so P(pharmacy | x) = σ(2f).

**Weak learner.** `fit_stump` searches every column, threshold and sign for the **minimum
weighted 0–1 error** (Bishop's definition), not a Gini split. It is checked against brute force
in the tests. On binary columns it is again a matrix product.

**On the pharmacy data** (figure U5_09): M = **812** rounds chosen by 5-fold CV error (CV
error 0.0222). Test F1 **0.9101**, precision **1.000** (0 false positives), recall 0.835,
ROC-AUC 0.9771. Only **94** distinct columns are used in 812 rounds: the same strong words come
back with new weights. First rounds: "pharmacy" (ε 0.065, α 2.66), "medicals" (ε 0.312),
"shop" (ε 0.331, *negative* evidence), "medicals" again (ε 0.394)… After the first few rounds εₘ
stays near 0.4: each new stump faces the weights the earlier ones could not fix (U5_09b).

## 3.6 Gradient boosting (Friedman 2001)

Forward stagewise additive modelling for **any differentiable loss** L: at round m fit a
regression tree to the negative gradient (the "pseudo-residuals") and add it with shrinkage ν:

| loss | pseudo-residual rₙ = −∂L/∂F | leaf value |
|---|---|---|
| squared error ½(y − F)² | y − F | mean of r in the leaf |
| logistic (deviance) | y − σ(F) | one Newton step: Σr / Σ p(1 − p) |

F₀ = ȳ (squared) or log-odds of ȳ (logistic). `test_gradient_boosting_matches_sklearn`: own =
scikit-learn to 1e-8 for both losses.

**Depth = order of interaction.** A depth-1 tree (stump) uses one variable, so boosted stumps are
an **additive model** (a GAM fitted by boosting). Depth d allows d-way interactions.

**On the match score** (grid: depth 1–4 × ν ∈ {0.03, 0.1, 0.3}, M ≤ 600 by grouped CV;
figure U5_04):

| depth | best grouped-CV MSE (over ν, M) |
|---|---|
| 1 (additive) | 232.86 |
| 2 | 182.41 |
| **3** | **165.48** (ν = 0.1, M = 35) |
| 4 | 171.30 |

- Test: depth 3, ν 0.1, M 35: R² **0.7910** (RMSE 15.99); own = scikit-learn to 2.8e-14.
- Boosted **stumps** (additive): R² 0.7751, about the same as the additive GAM (0.7731), as
  theory predicts. Allowing interactions (depth ≥ 2) cuts CV MSE by **29%**: E01 again.
- Smaller ν needs proportionally more rounds (ν 0.03 → M 115; ν 0.3 → M 10) for about the same CV
  error: shrinkage trades rounds for smoothness.

**Interpretation** (figure U5_05). Permutation importance on test (MSE increase when one column is
shuffled): `category_match` **1824**, `within_radius` 184, `n_categories` 23, `distance_km` 22,
others ≈ 0. The **2-D partial dependence** (category × radius) is 5.96 / 5.96 for the wrong
category and 52.09 / 76.85 for the right one: the model learned an interaction contrast of
**24.8** points, close to the 28.2 in the raw cell means (E01).

**On the pharmacy data** (logistic loss, ν 0.1, M ≤ 500 by 5-fold CV log loss): best depth 2,
M 245, CV log loss 0.0766 (depth 1: 0.0793, depth 3: 0.0778). Test F1 **0.9198**, log loss 0.0690.
scikit-learn with the same settings gets F1 0.9297. Many binary columns tie exactly on the split
gain, and the two implementations break those ties differently. Over 245 rounds the
probabilities drift apart (max |Δp| 0.36), while the quality is the same (log loss 0.0690 vs 0.0684).

---

# Part D: Additive models and Bayesian averaging

## 3.7 Generalized additive model

**Model.** y = α + Σⱼ fⱼ(xⱼ) + ε, each fⱼ centred (Σₙ fⱼ(xₙⱼ) = 0) so α is identifiable.

**Smooth terms = P-splines.** fⱼ(x) = Bⱼ(x)βⱼ with a cubic B-spline basis on 12 equal segments
(15 functions) and a roughness penalty λ‖D₂βⱼ‖² (second differences of neighbouring
coefficients). Binary inputs are linear terms.

**Penalised least squares**

  min ‖y − α − Σⱼ Bⱼβⱼ‖² + λ Σⱼ βⱼᵀ D₂ᵀD₂ βⱼ

solved two ways:
- **directly** (one ridge-type solve), which also gives the hat matrix H, the effective degrees of
  freedom edf = tr(H) and GCV = n·RSS/(n − edf)²;
- by **backfitting**: cycle fⱼ ← Sⱼ(y − α − Σ_{k≠j} f_k) with each term's own smoother
  Sⱼ = Bⱼ(BⱼᵀBⱼ + λD₂ᵀD₂)⁻¹Bⱼᵀ. This is block Gauss–Seidel on the same normal equations, so it
  converges to the direct solution: 75 sweeps, max |Δ| **3.7e-09**
  (`test_backfitting_converges_to_direct_gam_solution`).

λ is chosen by **grouped CV** (λ = 3.16). GCV would pick 0.01, because it assumes independent
rows and so does not see that pairs of the same buyer are alike.

| model | edf | test R² | test RMSE |
|---|---|---|---|
| GAM, additive | 12.78 | 0.7731 | 16.67 |
| **GAM + 2 interaction terms (GA²M)** | 16.67 | **0.8387** | **14.05** |

- Additive edf per term: s(distance_km) 5.78, s(n_categories) 2.00, each binary term 1.
  The distance curve (U5_06a) rises over the first ~5 km, then falls. Its wiggle around 30–35 km
  rests on few pairs and is probably noise. The n_categories curve (U5_06b) falls to a minimum
  at 6–7 categories: multi-category "everything" shops score lower. Above 7 there are only a
  handful of vendors, so the upturn is not reliable.
- The additive GAM beats OLS (0.773 vs 0.751) through the smooth distance curve, but it **cannot**
  represent E01. Adding exactly the two interactions the EDA found,
  `within_radius × category_match` and a separate distance curve for the right category
  (`s(distance_km) × category_match`, edf 4.19), gives **R² 0.8387, the best of all models in
  PBL-1, 2 and 3**. It is still fully interpretable: each term is one curve or one number.

## 3.8 Bayesian model averaging (Bishop 14.1)

  p(y | x, D) = Σ_M p(y | x, M, D) p(M | D),   p(M | D) ∝ p(D | M) p(M).

The models are all **2⁶ = 64** subsets of the six inputs (each with an intercept), with a uniform
prior. The evidence comes from the **Zellner g-prior**
β_M | σ² ~ N(0, g σ² (X_cᵀX_c)⁻¹), g = n = 360 (unit information). Its Bayes factor against the
intercept-only model has a closed form that depends only on R²:

  ln BF(M : M₀) = ((n − 1 − p_M)/2) ln(1 + g) − ((n − 1)/2) ln(1 + g(1 − R²_M)).

Given M, the posterior mean is β = g/(1 + g) · β̂_OLS (shrinkage) and the predictive is a
Student-t. BIC weights (−BIC/2) are computed as a second evidence approximation.

| rank | inputs | R² | p(M\|D) g-prior | p(M\|D) BIC |
|---|---|---|---|---|
| 1 | within_radius, category_match, keyword_in_title, n_categories | 0.7955 | **0.580** | 0.579 |
| 2 | + distance_km | 0.7979 | 0.241 | 0.249 |
| 3 | within_radius, category_match, n_categories | 0.7897 | 0.076 | 0.070 |
| 6 | all six (Phase 1 OLS) | 0.7983 | 0.019 | 0.019 |

- **Posterior inclusion probabilities:** within_radius 1.000, category_match 1.000,
  n_categories 1.000, keyword_in_title 0.880, **distance_km 0.298**, **is_chain 0.068**. The data
  does not need `is_chain` (PBL-1: t = 0.87), and once `within_radius` is in, raw distance
  adds little to a *linear* model.
- 6 of 64 models hold 99% of the mass. g-prior and BIC agree closely (n = 360 is large enough).
- Test: BMA R² **0.7492**, top model 0.7475, full OLS 0.7510: averaging over **linear** models
  cannot fix a **non-linear** truth. All 64 candidates share the same missing interaction.
- The **uniform** average of the 64 models is far worse (R² 0.5763): equal weights give the
  terrible models (no `category_match`) as much say as the good ones. BMA's weights are what
  make it work.
- BMA 95% predictive intervals cover **93.3%** of test scores (target 95%).

**BMA is not model combination** (figure U5_07b). Refitting on random subsets of requirements:

| training pairs | 30 | 60 | 120 | 240 | 360 |
|---|---|---|---|---|---|
| p(top model \| D) | 0.310 | 0.364 | 0.403 | 0.486 | 0.580 |
| effective number of models exp(H) | 11.2 | 8.6 | 6.3 | 4.4 | 3.4 |

As N grows, the posterior **concentrates** on one model. BMA's averaging expresses uncertainty
about *which single model* generated all the data, and it disappears with enough data. A
committee or mixture keeps combining models however much data there is (Bishop 14.1).

**Logistic BMA** over the 2¹⁰ = 1,024 subsets of PBL-1's 10-word model (the one where the MLE
exists), with BIC evidence: inclusion "shop" 1.000, "pharmacy" 1.000, "stores" 0.468,
"home" 0.384, others < 0.21. Test F1 0.5303 for BMA, top model and full 10-word MLE alike
(precision 1.000, recall 0.361); BMA has the best PR-AUC of the three (0.478 vs 0.470 / 0.432).
The model space is too weak to classify well (it excludes the strong words because the MLE
would not exist), and averaging inside it cannot add what no member has.

---

# Part E: Conditional mixture models (Bishop 14.5)

## 3.9 Mixture of linear regressions and mixture of experts

**Mixture of regressions:** p(t | x) = Σₖ πₖ N(t | wₖᵀφ(x), βₖ⁻¹). EM, as in PBL-2 but with a
regression in each component:
- E-step: γₙₖ ∝ πₖ N(tₙ | wₖᵀφₙ, βₖ⁻¹);
- M-step: πₖ = mean γₙₖ; wₖ = (ΦᵀRₖΦ)⁻¹ΦᵀRₖt (weighted least squares with Rₖ = diag γ·ₖ);
  βₖ⁻¹ = Σγₙₖ(tₙ − wₖᵀφₙ)² / Σγₙₖ, floored at 1 point² (see §1.7).
The log-likelihood never decreased in any run (figure U5_08a). A test recovers two planted lines.

| K | train ln p | BIC | **test ln p per pair** | test RMSE of the mean |
|---|---|---|---|---|
| 1 (= OLS) | −1485.1 | 3017.4 | −4.305 | 17.46 |
| 2 | −1411.3 | 2922.6 | −4.593 | 20.81 |
| 3 | −1286.2 | 2725.5 | **−3.813** | 19.25 |
| 4 | −1235.0 | **2676.0** | −4.324 | 19.26 |

With **input-independent** πₖ the predictive *mean* is one blended line, no better than OLS. The
gain is in the **distribution**: K = 3 has a better held-out likelihood than OLS (−3.81 vs −4.30
per pair). K = 2 finds a precise component (s.d. 4.4 points, within_radius −3.8) and a noisy one
(s.d. 14.8, within_radius **+28.7**): the two radius slopes of E01. Mean responsibility of the
noisy component is 0.92 in the right-category, inside-radius cell.

**Mixture of experts:** let the gate depend on x, π₁(x) = σ(vᵀx). The M-step for the gate is a
logistic regression with **soft** targets γₙ₁ (still concave; Newton/IRLS with a small L2).
Result: test R² **0.7916**, RMSE 15.97, test ln p per pair **−3.953** (OLS −4.305). The gate learns
the cells by itself: π₁ = 0.999 for wrong category outside radius, **0.018** for right category
inside radius. Expert 1 (s.d. 5.1) handles the "bad match" regime, expert 2 (s.d. 17.9) the "good
match" regime. That is E01 and E08, discovered without being told.

---

## 3.10 Summary table (test set; full lists in [unit5_numbers.md](unit5_numbers.md))

**Match score** (R², 90 test pairs, 6 unseen buyers):

| family | model | test R² |
|---|---|---|
| PBL-1 | OLS / kernel ridge RBF | 0.7510 / 0.8127 |
| tree | CART pruned (4 leaves) | 0.7700 |
| committee | bagged trees / random forest | 0.7742 / 0.7991 |
| boosting | boosted stumps / gradient boosting depth 3 | 0.7751 / 0.7910 |
| additive | GAM / **GA²M (GAM + 2 interactions)** | 0.7731 / **0.8387** |
| Bayesian | BMA over 64 linear models | 0.7492 |
| mixture | mixture of 2 linear experts | 0.7916 |
| learned combination | stacking (NNLS) / simple average | 0.8056 / 0.8025 |

**Pharmacy** (F1 at threshold 0.5, 923 test vendors): CART 0.911, bagging 0.896, random forest
0.925, AdaBoost 0.910 (precision 1.000), gradient boosting 0.920; PBL-1 L2 logistic 0.935.
Stacking, voting and significance tests are in §4.

---

## Viva questions on this section

**Q1. Derive the AdaBoost weight αₘ.**
Fix the earlier terms. E = Σₙ wₙ exp(−½αₘ tₙ yₘ(xₙ)) = e^{−α/2}·W_correct + e^{α/2}·W_wrong.
Setting ∂E/∂α = 0 gives e^{α} = W_correct / W_wrong = (1 − ε)/ε, so α = ln((1 − ε)/ε). A stump
with ε = 0.065 ("pharmacy") gets α = 2.66; one with ε = 0.4 gets 0.41.

**Q2. Why does a random forest beat bagging even though its trees are worse?**
Because the committee error depends on the correlation between members. On the score data the
RF trees have higher individual MSE (361.7 vs 321.7), but correlation 0.70 instead of 0.87, so the
average is better (245.9 vs 276.4). Random feature subsets de-correlate the trees.

**Q3. What does pruning optimise, and how is α chosen?**
R_α(T) = training error + α × leaves. Each α has one optimal subtree, and the subtrees are nested,
so one pass of weakest-link pruning gives all of them (144 here). α is chosen by grouped CV.
Min CV gave 4 leaves.

**Q4. How is gradient boosting "gradient descent"?**
In function space: F ← F − ν·∇L, where the gradient is evaluated at the training points
(pseudo-residuals) and approximated by a regression tree so that it generalises. For squared
error the negative gradient is just the residual y − F.

**Q5. Why are boosted stumps an additive model?**
Each stump depends on one variable, so the sum groups into Σⱼ fⱼ(xⱼ). That is the GAM form, and
the test R² is nearly the same (0.775 vs 0.773). Depth 2 or more allows interactions.

**Q6. Is BMA the same as averaging the models?**
No. It weights each model by its posterior probability. Uniform averaging of the 64 models gives
R² 0.576; BMA gives 0.749. And BMA's weights concentrate on one model as N grows
(effective number 11.2 at 30 pairs → 3.4 at 360). It is model **uncertainty**, not model
**combination**.

**Q7. Why is backfitting guaranteed to converge here?**
With symmetric, linear smoothers (penalised regression splines) backfitting is block
Gauss–Seidel on the normal equations of one convex penalised least-squares problem, which has a
unique minimiser (a tiny ridge removes the flat direction from centring). The test checks it
reaches the direct solve.

**Q8. Your best model is a GAM with two interactions, not an ensemble. Does that undermine
Unit V?**
No, it is the main lesson. The trees and boosting **discovered** the interaction (2-D partial
dependence contrast 24.8). Writing it into an additive model then gives a model that is both more
accurate and fully interpretable. Ensembles are also the tool for finding what a simple model
is missing.

**Q9. How did you check your implementations?**
`tests/test_ensemble.py`, 10 tests: CART = scikit-learn (regression, classification, binary
path), pruning path = scikit-learn, stump = brute force, AdaBoost's exponential loss = the product
bound, gradient boosting = scikit-learn (both losses), committee error ≤ average error,
backfitting = direct solve, BMA's full model = OLS and the g-prior Bayes factor formula, mixture
EM monotone and recovers planted lines.
