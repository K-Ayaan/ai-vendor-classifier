# 5. Result Interpretation & Presentation (4 marks)

> What the examiner checks: can you say what the numbers **mean** in plain words, explain
> **where and why** the model fails, state limitations honestly, and present it all in a clear
> order?

---

## 5.1 Interpreting the regression model

**Coefficients in plain words** (all else held fixed):

| term | β̂ | meaning |
|---|---|---|
| category_match | +59.44 | the right kind of shop is worth about 59 points, the dominant factor (t = 35.2) |
| within_radius | +18.14 | being inside the buyer's limit adds about 18 points |
| keyword_in_title | +15.47 | matching the preference adds about 15 points (least precise: SE 4.91, only 12 pairs have it) |
| n_categories | −3.24 | each extra listed category costs about 3 points |
| distance_km | −0.157 | each km costs about 0.16 points (small: the radius flag already captures most of it) |
| is_chain | +3.47 | not distinguishable from 0 (t = 0.87) |
| intercept | +0.30 | an anchor, not a real vendor (no vendor has n_categories = 0) |

**Worked predictions** (n_categories = 1, not a chain):

| vendor | ŷ | rubric anchor |
|---|---|---|
| right category, 1 km, inside radius, keyword match | 89.94 | 100 |
| right category, 1 km, inside radius, no keyword | 74.48 | 75 |
| right category, 10 km, outside radius | 54.93 | 50 |
| wrong category, 1 km, inside radius | 15.04 | 0–25 |
| wrong category, 30 km, outside radius | −7.65 | 0 (impossible negative) |

**So the model has learned the rubric's structure.** That is expected, because the scores were
produced against that rubric, so **a high R² means "learns the rubric well", not "is a perfect
real-world matcher".**

**R² 0.751 on unseen requirements** = the model explains 75% of the variance of scores for
buyers it never saw. The typical miss is ~14 points (MAE 13.98), and big misses push RMSE to 17.46.

**Train 0.798 vs test 0.751:** a small gap, so no serious overfitting (graph `07`).

---

## 5.2 What the residuals say (graphs 02, 03, 04)

| Observation | Meaning |
|---|---|
| Residuals form diagonal bands (graph 02) | the target is lumpy (41.9% on anchors), so for a fixed category/radius combination the residuals line up |
| Heavy tails, excess kurtosis 2.22, Shapiro-Wilk p = 3.6e-08 (graphs 03, 04) | Gaussian errors are only approximate, so SEs and p-values are approximate |
| 61 train / 14 test predictions below 0 | a linear model is unbounded; the prototype stores the raw value and clips only for display |
| Largest negative residuals | vendors listed in the category whose title shows another kind of shop (e.g. a handicraft shop listed under fruits): **category-label noise**, which the judge caught and the features can't |

---

## 5.3 Error analysis on held-out requirements (from `reports/phase2_results.md`)

The 5 largest gaps between the shown score and the rubric score, with the reason computed from
the real contributions:

| vendor (requirement) | rubric | shown | real reason |
|---|---|---|---|
| D-Mart (kirana_provisions) | 92 | 45.2 | listed in 10 categories → n_categories adds **−32.40**; a big supermarket is actually a great fit |
| Home Needs Super Market | 92 | 45.3 | same: 10 categories, −32.40 |
| Nilgiris Canopy | 92 | 48.7 | 9 categories, −29.16 |
| "Fruits" (street_markets, keyword "fruit") | 40 | 0.0 (raw −6.65) | wrong category and 23.6 km away; also "fruits" ≠ the token "fruit", so the keyword flag is 0 |
| V style & minimart (street_markets) | 30 | 64.4 | listed under street_markets (+59.44) but it is a mini-mart/clothing shop: **noisy label** |

**Two lessons:**
1. `n_categories` is a **linear** penalty: right for mislabelled "everything" shops, wrong for big
   supermarkets. A non-linear model or an interaction (category × n_categories) would fix it.
2. Exact token matching is brittle (fruit/fruits). Stemming would help.

---

## 5.4 Interpreting the classifier

- **Confusion matrix (L2, test):** TN 825, FP 1, FN 11, TP 86.
  - Precision 86/87 = **0.9885**: when it says "pharmacy", it is almost always right.
  - Recall 86/97 = **0.8866**: it finds 89% of (file-labelled) pharmacies.
- **ROC-AUC 0.9842:** a random pharmacy outranks a random non-pharmacy 98.4% of the time.
- **PR-AUC 0.9465** against a chance level of 0.1051 (the prevalence).
- **Some "errors" are label errors.** A blind check of 200 test vendors (title and address
  only) agreed with the file label on 93.5% (98.59% re-weighted to the test mix, κ 0.9204).
  Every disagreement was a file-"pharmacy" that is not one (a chemistry academy, hospitals, a
  medical college, supermarkets). Against those audit labels, the L2 model's re-weighted F1 is
  **0.9647**, higher than against the file labels. Part of the "missed" pharmacies were never
  pharmacies.
- **Biggest words** (E07): medicals, medical, apollo, medplus, pharmacy push towards
  pharmacy; market, ration, xerox push away.

---

## 5.5 Key takeaways (the five sentences to remember)

1. **Least squares is Gaussian MLE.** It explains 75% of the score variance on new buyers, and
   category match dominates (+59 points).
2. **The logistic MLE does not exist on this data** because of separation. A small model gives
   a true MLE; L2 (MAP) gives the useful classifier (F1 0.935).
3. **Interactions matter.** Kernel ridge improves test R² 0.751 → 0.813, because a right
   category inside the radius is worth more than the sum of its parts.
4. **Discriminative beats generative** here (logistic/SVM 0.935 > naive Bayes 0.910 > LDA 0.519).
5. **Label noise is the main limit.** The worst errors are mislabelled categories, and the
   labels themselves come from a rubric score and a file membership, not verified truth.

---

## 5.6 Limitations (say them before you're asked)

| Limitation | Effect | Next step |
|---|---|---|
| Scores come from an AI judge against a rubric, not humans | R² measures agreement with the rubric | collect human scores; report agreement |
| Weak category labels | wrong-shop vendors get +59 points | classifier check for all categories; title-based category model |
| Requirements are generated, not real buyers | realistic but synthetic | log real requests |
| Single snapshot | no opening hours/stock; no temporal test | repeated scrapes |
| Linear, unbounded score | negatives (14 test); additive effects | kernel/interaction terms; bounded link (e.g. logistic on score/100) |
| 6 test requirements | small differences are noise | more requirements; repeated grouped CV |

---

## 5.7 How to present (about 10 minutes, in this order)

| Min | Slide | Show | Say |
|---|---|---|---|
| 0–1 | Problem | §1.1 | "Score and rank vendors for a buyer's need; check category labels." |
| 1–2 | Data | E01, E04 | "5,439 records → 4,612 vendors, 23 categories, one snapshot; no labels, so I built requirements and a rubric." |
| 2–4 | EDA | E07, E10, E09 | "One-class words predict separation; category × radius interact; the target is lumpy." |
| 4–6 | Linear MLE | graphs 05, 06 | derive ℓ, RSS = MLE, β̂; σ̂² 224.28 vs s² 228.72; ℓ_max −1485.14 |
| 6–7 | Diagnostics | graphs 01, 02, 04 | "R² 0.80/0.75; heavy tails; negative predictions." |
| 7–8 | Logistic | graphs 12, 11, 09 | "Why not linear; IRLS; MLE doesn't exist; small MLE vs L2 MAP; F1 0.935." |
| 8–9 | Comparison | C01, C02, graph 13 | baselines → kernels best; discriminative wins; rule baseline 0.891 |
| 9–10 | Interpretation & limits | §5.3, §5.6 | worst errors = label noise; honest limitations; next steps |

**Presentation tips**
- Lead every slide with **the claim**, then the evidence (number + plot).
- When you show a number, say **what it means** ("0.75 = 75% of variance on new buyers").
- Mention **one thing that didn't work** (ridge = OLS, Bayesian logistic less sharp). Examiners
  trust honest results.
- Keep `notebooks/03_linear_mle.ipynb` open: each derivation is followed by the cell that proves
  it with real numbers.

---

## Viva questions on this section

**Q1. Your R² is 0.75. Is the model good?**
Relative to the baseline (R² 0, RMSE 34.98) it is far better, and it generalises to new
buyers. But it learns a rubric score, so it is "good at reproducing the rubric"; real-world
quality needs human-labelled data.

**Q2. Interpret β for category_match.**
Holding the other features fixed, a vendor listed in the needed category scores 59.44 points
higher on average. It is the largest and most certain effect (t = 35.2).

**Q3. Why do some predictions go below zero?**
Linear regression is unbounded: a wrong-category vendor 30 km away gets 0.30 − 0.157·30 − 3.24 ≈
−7.65. The rubric floor is 0. In the prototype we keep the raw value and clip only for display.

**Q4. What is the single biggest weakness?**
Label noise in categories: a vendor listed under the wrong category gets +59 points. The worst
errors (mini-mart under street_markets) come from this.

**Q5. If you had one more week, what would you do?**
Collect human scores for the 450 pairs to validate the target; add a category × n_categories
interaction (fixes the supermarket under-scoring); repeat grouped CV for confidence intervals.

**Q6. What does PR-AUC 0.9465 mean, and why report it?**
The area under the precision–recall curve. With 10.5% positives, its chance level is 0.105, not
0.5, so it is more honest than ROC-AUC for a rare class.
