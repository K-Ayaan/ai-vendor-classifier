# 5. Result Interpretation & Presentation (4 marks)

> What the examiner checks: can you say what the numbers **mean** in plain words, explain
> **where and why** the methods fail, state limitations honestly, and present it all in a clear
> order?

---

## 5.1 What the combined score models learned (in plain words)

**The rule every good model agrees on** (pruned CART, U5_01; partial dependence, U5_05):

1. *Is it the right kind of shop?* If not, the score is about 3–9 whatever else is true. This
   one question explains most of the variance: `category_match` importance 0.68 (tree), 0.69
   (forest); permutation importance 1,824 vs 184 for the next input.
2. *If it is the right kind: is it inside the buyer's radius?* Inside ≈ 77–84, outside ≈ 41–53.
3. *Then fine detail:* farther away is a bit worse (the additive GAM curve falls about 8 points
   from 5 km to 25 km), and vendors listed under many categories score lower ("everything" shops
   are poorer matches).

**What that means for the agent:** filter by category first, then by radius, then rank by
distance and specialisation. That is the human rubric (`RUBRIC.md`) recovered from data. The
PBL-1 linear model had to express "radius matters only for the right category" with one shared
coefficient, and so got both cells wrong (E01).

**Why the GAM with two interactions is the best model (R² 0.839):**
- it has the **shape** the data needs: a step for the radius **inside** the right category
  (`within_radius × category_match`) and a separate distance curve for right-category vendors
  (`s(distance) × category_match`, edf 4.2);
- it has **few effective parameters** (edf 16.7 for 360 rows), so it does not overfit like a
  258-leaf tree (train R² 1.000, test 0.735);
- the trees told us **which** interactions to add (2-D partial dependence contrast 24.8 vs 28.2
  in the data). Ensembles worked here as a **discovery tool**, and the additive model as the
  **final, explainable** model.

**Why the tree ensembles did not beat kernel ridge:** with 6 inputs, 4 of them binary, there are
only 2⁴ = 16 combinations of the binary inputs. An RBF kernel already fits a smooth surface over
them. Trees approximate the distance effect with steps, so they need more data than 360 pairs to
be as smooth.

## 5.2 Interpreting the Bayesian model average

- **Inclusion probabilities** are the clearest single output of PBL-3 for a non-expert: the data
  is certain it needs `category_match`, `within_radius`, `n_categories` (all ≈ 1.000) and fairly
  sure about `keyword_in_title` (0.880). It doubts `distance_km` (0.298) *in a linear model* and
  does not need `is_chain` (0.068).
- That does **not** mean distance is irrelevant. The trees and GAM use it (forest importance 0.17).
  It means a straight-line distance effect adds little once the radius flag is in. BMA answers
  "which inputs, given this model family", never "is the family right".
- BMA's 95% intervals cover 93.3% of test scores but are **60 points wide**: honest about
  uncertainty, too wide for ranking. The uncertainty is mostly the mis-specification that GA²M
  removes.
- **As data grows the posterior concentrates** (effective number of models 11.2 → 3.4 from 30 to 360
  pairs). With a few thousand pairs BMA would be one model. Averaging under model uncertainty is
  not a way to combine models.

## 5.3 Interpreting the pharmacy ensembles

- Every ensemble agrees with PBL-1 on the evidence: **"pharmacy", "medicals", "medical",
  "medplus", "apollo"** carry almost all the weight (RF importance 0.33 / 0.25 / 0.09 / 0.04 / 0.04).
  AdaBoost's first stumps also use "shop", "market", "stall" as **negative** evidence.
- AdaBoost needs **812 rounds but only 94 distinct words**: it keeps re-weighting the same strong
  words to fix the remaining hard vendors.
- Because the decision is close to **additive in the words**, a penalised linear model (PBL-1)
  is the right shape, and combining trees adds variance without fixing bias. The opposite of the
  score model.

## 5.4 Error analysis

**Match score: the hardest buyer, R18** (RMSE 19–25 for every model, C03). R18 is a **kirana
(provisions)** request, and every model makes its largest errors on the **same** pairs:

| pair | vendor | categories listed | judge | OLS error | GA²M error | forest error |
|---|---|---|---|---|---|---|
| P258 | D-Mart - Chengalpattu | 10 | 92 | −46.8 | −31.1 | −38.4 |
| P269 | Home Needs Super Market | 10 | 92 | −46.7 | −31.2 | −37.2 |
| P262 | Nilgiris Canopy | 9 | 92 | −43.3 | −33.7 | −43.1 |

These are large supermarkets: the right category, inside the radius, rated excellent by the
judge. All models **under-predict by 31–47 points**, because they learned from training that
vendors listed under many categories score lower (the n_categories curve, U5_06b). That holds
for "everything" shops in most categories, but for a provisions need a big supermarket **is** a
top match. The penalty should depend on the **needed** category, an interaction between
`n_categories` and the requested category that no model was given. Other causes of error:
- the keyword match is exact ("fruits" ≠ "fruit", PBL-1 §1.2);
- AI-judged scores carry their own noise. The blind re-score agreement was r = 0.976
  (PBL-1), which puts a ceiling on the reachable R².

**Pharmacy: the 10 errors shared by all nine strong classifiers (C06):**

| type | count | example | what would fix it |
|---|---|---|---|
| **wrong weak label** (not a pharmacy, filed under pharmacy) | **8** | More Supermarket, ANBU CHEMISTRY ACADEMY, MRS Surgicals | clean the labels, not the model. The models are right |
| **unseen word** (a real pharmacy using a rare word) | 2 | NEW PHARMA WORLD, PRABAL PHARMA ("pharma" appears in < 5 training titles) | sub-word features (character n-grams, "pharm*"), or a lower min_df |

False positives differ between models (0 for AdaBoost, up to 9 for bagged trees). Typical cases
are names containing pharmacy-like words in other contexts: *Sweet Truth Potheri* ("Potheri" is a
locality with many pharmacies), *The Buffet Square By SRM Hotels* ("SRM" is a hospital chain
name), *AUROPET CLINIC & PETSTORE* ("clinic"). These are combinations a human reads at once and a
bag of words cannot.

## 5.5 Key takeaways (the five sentences to remember)

1. **Combining models helps exactly when one model cannot represent the truth.** For the match
   score (an interaction and a step), trees, forests, boosting, stacking and a mixture of experts
   all beat OLS; for the pharmacy label (nearly linear in words) nothing beats L2 logistic.
2. **Committees work by de-correlating errors:** the random forest's trees are individually worse
   than bagged trees (MSE 362 vs 322) but less correlated (ρ 0.70 vs 0.87), so the forest is better
   (246 vs 276).
3. **Boosting depth is the order of interaction:** stumps = an additive model (R² 0.775 ≈ GAM
   0.773); depth 3 cuts CV error 29% because it can model category × radius.
4. **BMA is model uncertainty, not model combination:** over linear models it equals OLS (ΔMSE
   +2.2, CI includes 0), and its weights concentrate as data grows.
5. **The best model is simple and informed by the ensembles:** a GAM with the two interactions
   the trees revealed, test R² 0.839 (from 0.751), and every term is a readable curve.

## 5.6 Limitations (say them before you're asked)

| limitation | effect | what would fix it |
|---|---|---|
| Only **6 test buyers** | wide intervals; most model differences are not significant | more requirements, or nested grouped CV over all 30 |
| **AI-judged** targets and labels | models learn the rubric and the judge, not real buyers | human scores and labels |
| GA²M's interaction terms were **chosen by me from the EDA** | uses human insight; would need re-checking on new data | select terms automatically (e.g. by CV over candidate pairs) |
| 360 training pairs | ensembles' advantages (variance reduction) show up more with more data | more pairs |
| Only **6 coarse features** for the score | every model shares the R18-type errors (big supermarkets under-scored by 31–47 points) | the needed category as an input (so the n_categories effect can depend on it), fuzzy keyword match, road distance |
| Bag-of-words pharmacy features | rare words ("pharma") are invisible; context is ignored | character n-grams or sub-word features |
| Mixture-model variance floor (1 point) is a choice | changes the likelihood numbers | a prior on the variances (MAP-EM) |
| Training time not benchmarked | can't trade accuracy against cost precisely | time each fit |

## 5.7 How to present (about 10 minutes, in this order)

| min | slide / figure | say |
|---|---|---|
| 0–1 | the problem | "PBL-1 used one model per task. Unit V asks: can a combination of models do better, and when?" |
| 1–3 | **E01**, **E05**, **E06** | "The score has an interaction: radius +31 for the right shop, +3 for the wrong one. A single tree is unstable (s.d. 7.1 vs 1.9). Each word alone is a weak learner." |
| 3–4 | **U5_01** | "A pruned tree with 4 leaves already beats OLS. It is the rubric as rules." |
| 4–5 | **U5_02**, **U5_03** | "Forests beat bagging because they de-correlate trees. Bias–variance: OLS has bias, a deep tree variance, forests and boosting have neither." |
| 5–6 | **U5_04**, **U5_05** | "Gradient boosting: depth 1 is additive, depth 3 finds the interaction. Partial dependence shows it learned +25 points." |
| 6–7 | **U5_06** | "So I put the two interactions into a GAM: R² 0.839, the best model, and fully readable." |
| 7–8 | **U5_07** | "BMA over 64 linear models: tells which inputs matter, but equals OLS, and concentrates as N grows. BMA is not combination." |
| 8–9 | **C02**, **C04** | "Paired tests: interaction-capable models beat OLS. For pharmacies nothing beats L2 logistic, and McNemar shows no significant gain." |
| 9–10 | **C06**, §5.5 | "Every model makes the same 10 pharmacy mistakes, and 8 are wrong labels. Combining helps only when members make different errors. Takeaways." |

**Have open:** `src/ensemble/boosting.py` (AdaBoost, 30 lines), `tree.py · pruning_path`,
[unit5_numbers.md](../03_unit_topics_implementation/unit5_numbers.md), `tests/test_ensemble.py`.

---

## Viva questions on this section

**Q1. In one sentence: when should the agent use an ensemble?**
When the relationship is non-linear or has interactions that a simple model cannot represent, as
for the match score. When the simple model is already the right shape, as for the pharmacy label,
an ensemble only adds cost.

**Q2. Would you deploy the GAM or the random forest?**
The GAM with interactions: it is the most accurate (R² 0.839), it is fast, and each term can be
shown to a user ("this vendor lost 8 points for distance"). The forest is a good check that
nothing important is missing.

**Q3. Is R² 0.839 good?**
It is relative to an AI-judged rubric whose own blind re-score correlation is 0.976 (r² ≈ 0.95),
so there is still room. But the interval [0.75, 0.91] is wide with 6 test buyers, so claim
"clearly better than OLS, probably better than kernel ridge", not a precise number.

**Q4. Why didn't stacking win if it combines the best models?**
It did beat OLS (ΔMSE −67), but against kernel ridge its gain is within the noise. Its members'
errors are correlated, and learned weights add their own variance with 360 rows. Stacking helps
most with many rows and diverse, strong members.

**Q5. What is the most surprising result?**
That 8 of the 10 pharmacy "errors" all models share are wrong labels. Model agreement turned out
to be a label-cleaning tool: when nine different models confidently disagree with the label, check
the label.

**Q6. What would you do next?**
Score more requirements (to tighten the intervals), give the models the needed category so the
"many categories" penalty can depend on it (to fix R18, where big supermarkets are under-scored
by 31–47 points), add fuzzy keyword matching, relabel the vendors all models disagree with, and
let the GA²M choose its interaction terms by grouped CV.
