# 1. Problem Understanding & Dataset Familiarization (4 marks)

> What the examiner checks: can you state the problem precisely, explain why ML fits it,
> describe the raw data field by field, and say what is missing or unreliable in it?

---

## 1.1 The problem in one paragraph

A **buyer** at a location needs a **local vendor** of a certain kind (a pharmacy, a vegetable
seller, a hardware shop…), within a **maximum distance**, sometimes with a **preference**
("wholesale", "organic", "24-hour"). Thousands of shops exist in the area, and their listings are
noisy. The project builds the statistical core of an **intelligent vendor-matching agent**:

1. **Regression:** predict a **match score (0–100)** for any (requirement, vendor) pair, so
   vendors can be ranked.
2. **Classification:** predict whether a vendor **really is a pharmacy** from its name and
   location, so a wrong listing can be flagged.

Both models are fitted by **maximum likelihood estimation (MLE)**, the core of Unit II.

## 1.2 Why this is a machine-learning problem (and not just a filter)

| A plain filter would… | …but in reality |
|---|---|
| Keep vendors listed in the right category | Category labels are noisy: a toll plaza is listed under street markets, pet shops under kirana_provisions |
| Keep vendors within the radius | A vendor 3.1 km away may still be a good answer for a 3 km limit; a scored ranking expresses "how good" |
| Match the keyword exactly | "fruits" ≠ "fruit": exact matching misses near-matches |

So we need a **graded score** learned from examples, plus a **classifier** that can check a
category label from the text. That is regression and classification.

## 1.3 Inputs, outputs and success criteria

| | Regression (match score) | Classification (is pharmacy) |
|---|---|---|
| Unit of analysis | one requirement–vendor pair | one vendor |
| Rows | 450 pairs (30 requirements × 15 candidates) | 4,612 vendors |
| Inputs (X) | 6 pair features + intercept | 421 title words + 20 pincodes + intercept = 442 columns |
| Output (y) | `expected_score`, integer 0–100 | `is_pharmacy`, 0/1 |
| Split | by requirement: 24 train / 6 test (360 / 90 pairs) | stratified 80/20: 3,689 / 923 vendors |
| Success measure | R², RMSE, MAE on **unseen requirements** | precision, recall, F1, ROC-AUC, PR-AUC vs the majority baseline |

## 1.4 The dataset, field by field

**Source:** 23 JSON files in `Dataset/` (copied byte-for-byte to `data/raw/`), one per retail
category, from a single Google Maps snapshot taken on **2026-09-27**, covering South Chennai
to Chengalpattu (about 45.5 km north–south × 40.8 km east–west).

Each file looks like this:

```json
{ "category": "pharmacy", "count": 486, "outside": 64, "updated_at_utc": "2026-09-27T12:23:25+00:00",
  "points": [ { "id": "chij...", "title": "Sri Balaji Medicals", "address": "..., Tamil Nadu 603002",
                "lat": 12.68, "lon": 79.97 }, ... ] }
```

| Field | Meaning | What we found (audit) | How we use it |
|---|---|---|---|
| `category` | the file's category | 23 categories | becomes the vendor's (multi-)label |
| `count` | header count of points | equals `len(points)` in all 23 files | **file metadata only, never a feature** |
| `outside` | unknown header metadata | 18–104 per file, meaning undocumented | **never used** (we don't use what we can't explain) |
| `updated_at_utc` | scrape time | one date for all files (2026-09-27) | proves it's a single snapshot, so no time-based split |
| `id` | place id | lowercase (cannot be looked up again) | kept as `source_place_id`; vendors get `V0001`–`V4612` |
| `title` | shop name | never null; 36 titles contain Tamil letters | the **only text**: classifier words, keyword, chain flag |
| `address` | free text | 1 empty; 4,611/4,612 contain a 6-digit pincode | pincode + locality extracted |
| `lat`, `lon` | coordinates | never null; lat 12.651–13.060, lon 79.905–80.281 | haversine distance to the buyer |

**Counts:** 5,439 point records → **4,612 unique vendors**. The 827 extra records are the same
shop appearing in several category files (title and lat/lon always identical across files: 0
conflicts). So we **merge by id**, and a vendor's `categories` is a **set** (88.77% have one
category; 3 vendors have 10).

## 1.5 What the dataset does NOT contain (and what we did about it)

| Missing | Consequence | Our answer |
|---|---|---|
| Buyer requirements | nothing to match against | **generated 30 requirements** (`src/make_requirements.py`): every category once, plus 7 random repeats; buyer location copied from a random "anchor" vendor; radius from {1, 3, 5, 10} km; optional keyword from an allow-list |
| Match scores | no regression target | **450 pairs scored against a written rubric** (`data/annotation/RUBRIC.md`, anchors 0/25/50/75/100). The scoring was done by an AI judge (Claude Code), one requirement at a time, reading title, address and categories, with a rationale per pair (`ai_scores.csv`). Consistency was checked by a **blind re-score** of 45 pairs: r = 0.9762, mean abs. difference 4.133 |
| True "is pharmacy" label | only file membership | used as a **weak label**; 200 test vendors were re-labelled by the AI judge from title/address: agreement 93.50% (98.59% re-weighted), Cohen's κ 0.9204 |
| Opening hours, stock, prices, ratings | can't model availability/quality | stated as a limitation |
| Timestamps per record | no temporal validation | single snapshot, stated as a limitation |

> **Say this confidently in the viva:** "No labelled data existed, so I designed the
> requirements, wrote a scoring rubric, and had pairs scored against it by an AI judge. I
> measured its consistency with a blind re-score (r = 0.976). These are not human ground truth,
> so a high R² shows the model learned the rubric, not that it is a perfect real-world matcher."

## 1.6 How candidates were chosen for each requirement (sampling design)

For each requirement, 15 candidates in 4 **strata**, so the scores cover the whole range:

| Stratum | n | How picked | Why |
|---|---|---|---|
| near_in_category | 5 | the 5 nearest vendors in the category | likely good matches |
| far_in_category | 4 | random in-category vendors **beyond** the radius | right shop, too far |
| near_other_category | 3 | the 3 nearest vendors of other categories | close but wrong shop |
| random | 3 | random vendors | background |

The 15 rows were **shuffled** and the stratum hidden from the scorer, so it could not infer the
answer from the order.

## 1.7 Assumptions and constraints (know these)

- **Straight-line distance** (haversine, Earth radius in `preprocessing.py`), not road distance.
- **Category = file membership** → noisy (weak supervision).
- **`is_chain`** = cleaned title shared by ≥ 3 vendors (234 vendors). Generic names like "ration
  shop" (39) and "fish market" (9) also pass, so it is not true chain detection.
- **Reproducibility:** seed 42 everywhere; raw data read-only (SHA-256 verified); every number
  in the reports is produced by code.
- **No leakage:** vocabulary, pincode list, CV and scaling are fitted on the training split only;
  regression is split **by requirement** so a test buyer is never seen in training.

## 1.8 Glossary (quick definitions for the viva)

| Term | One-line meaning |
|---|---|
| Requirement | (needed category, buyer lat/lon, max distance, optional keyword) |
| Pair | one requirement + one candidate vendor |
| Weak label | a label from a noisy proxy (file membership), not verified truth |
| Haversine | great-circle distance between two lat/lon points |
| Stratified split | train/test keep the same class ratio (10.5% pharmacies in both) |
| Group split | all rows of one requirement go to the same side |
| MLE | parameters that make the observed data most probable |

---

## Viva questions on this section

**Q1. State your problem in two sentences.**
Given a buyer's need, location, distance limit and preference, score every nearby vendor 0–100
and rank them; separately, classify whether a vendor is really a pharmacy. Both are fitted by
maximum likelihood (linear and logistic regression).

**Q2. Where does the data come from and how big is it?**
23 Google Maps category files from one snapshot (2026-09-27): 5,439 records, 4,612 unique
vendors, 23 categories, 106 pincodes.

**Q3. Why 4,612 and not 5,439?**
827 records are the same shop listed in more than one category file. Same id, same title, same
coordinates (0 conflicts), so we merge them and keep the set of categories.

**Q4. What are `count` and `outside`, and why didn't you use them?**
`count` equals the number of points (verified in all 23 files). `outside` is undocumented. We
never use a field we cannot explain, and neither is a property of a vendor.

**Q5. Your data had no labels. How did you get a target?**
I generated 30 realistic requirements, sampled 15 candidates each by stratum, wrote a 0–100
rubric, and had the 450 pairs scored against it by an AI judge, with a written rationale per
pair. A blind re-score of 45 pairs gave r = 0.976. I report them as AI-judged labels, not ground
truth.

**Q6. Why is the pharmacy label "weak"?**
It comes from which file a vendor is in. A check of 200 test vendors found mislabelled ones (a
chemistry academy, hospitals, supermarkets); agreement with the file label was 93.5% on the sample.

**Q7. Why split regression by requirement?**
The 15 pairs of one requirement share a buyer and a radius. A random split would leak that buyer
into training. At deployment, requirements are new, so the test should be too.

**Q8. What would you need to make this production-grade?**
Human-labelled match scores, real buyer requirements, opening hours and stock, road distance, and
repeated snapshots for temporal validation.
