# Annotation rubric: requirement-vendor match score (0-100)

Each row pairs a **requirement** (a buyer at a location who needs a vendor of
`needed_category`, within `max_distance_km`, optionally preferring
`preference_keyword`) with one **candidate vendor**. Score how well the vendor
meets the requirement.

## Anchors

| Score | Meaning |
|---|---|
| 0 | Unrelated vendor: wrong category and of no use for this need |
| 25 | Wrong category, but could partly help (e.g. a supermarket for a vegetables need) |
| 50 | Right category, poor fit: outside `max_distance_km` or does not match the preference keyword |
| 75 | Right category and within `max_distance_km` |
| 100 | Right category, within `max_distance_km`, and matches `preference_keyword` |

Use any whole number from 0 to 100. Scores between anchors are allowed (e.g. 60
for a right-category vendor just outside the radius).

When the requirement has **no preference keyword**, a right-category vendor
within the radius can score 100.

## How to judge

- **Category:** judge from the vendor title, address and `vendor_categories`.
  The listed categories are noisy (e.g. pet shops listed under
  kirana_provisions), so trust what the title tells you when they conflict.
- **Distance:** `distance_km` is the straight-line distance from the buyer.
  Compare it with `max_distance_km`.
- **Preference:** does the vendor plausibly meet the keyword (e.g. "wholesale",
  "24" for 24-hour)? It does not have to appear word-for-word in the title.

## Rules

1. Two raters score **independently**. Do not look at or discuss the other
   rater's scores until both are finished.
2. Rater 1 fills only `rater_1_score`; rater 2 fills only `rater_2_score`.
3. Use `notes` for anything unclear (e.g. "title suggests closed", "not sure
   it is a pharmacy"). Do not change any other column.
4. Do not sort, delete or add rows.
5. Save the file as **CSV (UTF-8)** with the same name.
