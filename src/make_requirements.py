"""Step 3 - generate requirements and the human annotation pack.

Usage: python src/make_requirements.py
Reads data/processed/vendors.csv and writes:
  data/processed/requirements.csv     30 requirements
  data/processed/pairs.csv            450 requirement-vendor pairs + features + stratum
  data/annotation/annotation_sheet.csv  sheet for two human raters (scores left blank)
  data/annotation/RUBRIC.md           0-100 scoring rubric
No scores or labels are generated here. An existing annotation sheet is never overwritten:
it is only checked against the regenerated pack (rater columns ignored).
"""
import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching.features import FEATURES, pair_features, title_tokens  # noqa: E402
from preprocessing import haversine_km  # noqa: E402

SEED = 42
VENDORS = Path("data/processed/vendors.csv")
OUT_REQ = Path("data/processed/requirements.csv")
OUT_PAIRS = Path("data/processed/pairs.csv")
OUT_SHEET = Path("data/annotation/annotation_sheet.csv")
OUT_RUBRIC = Path("data/annotation/RUBRIC.md")

N_REQUIREMENTS = 30
RADII_KM = [1, 3, 5, 10]
KEYWORD_PROB = 0.5           # preference keyword is optional
KEYWORD_MIN_DF = 3           # used by at least 3 vendors in the category
KEYWORD_MAX_SHARE = 0.25     # but by at most 25% of them (else it is just the category name)
# Frequent title tokens include brand/person names ("balaji", "poorvika"), place names
# ("chennai") and filler ("all", "pvt"), which are not buyer preferences. A keyword must
# therefore be BOTH a frequent token in the category AND one of these product/service
# descriptor words (a manual judgement, written out here so it can be reviewed).
KEYWORD_ALLOWLIST = {
    "wholesale", "organic", "24", "fresh", "farm", "farmers", "supermarket", "departmental",
    "department", "textiles", "textile", "clothing", "fashion", "shoes", "shoe", "spares",
    "spare", "wheeler", "bike", "car", "service", "cake", "cakes", "snacks", "sweet", "bakes",
    "cafe", "cosmetics", "salon", "natural", "naturals", "parlour", "computers", "computer",
    "digital", "appliances", "electrical", "electronics", "flowers", "bouquet", "decoration",
    "nursery", "garland", "restaurant", "tiffen", "mess", "fruit", "fruits", "vegetable",
    "vegetables", "plumbing", "tiles", "furniture", "plastics", "plastic", "decor", "pet",
    "pets", "aquarium", "provision", "mutton", "beef", "proteins", "gadgets", "accessories",
    "eyeglasses", "lens", "eye", "clinic", "hospital", "surgical", "book", "books", "fancy",
    "xerox", "fish", "government",
}
# candidates per requirement: (stratum, n)
STRATA = [("near_in_category", 5), ("far_in_category", 4), ("near_other_category", 3),
          ("random", 3)]

RUBRIC = """# Annotation rubric: requirement-vendor match score (0-100)

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
"""


def category_keywords(vendors, category):
    """Allow-listed descriptor words that are frequent (but not near-universal) title tokens
    in the category. Returned sorted by frequency."""
    in_cat = vendors[vendors.category_set.map(lambda s: category in s)]
    df = Counter(tok for t in in_cat.title for tok in title_tokens(t))
    max_df = KEYWORD_MAX_SHARE * len(in_cat)
    category_stems = {w.rstrip("s") for w in category.split("_")}  # "fruits" -> "fruit"
    ok = [(tok, n) for tok, n in df.items()
          if KEYWORD_MIN_DF <= n <= max_df and tok in KEYWORD_ALLOWLIST
          and tok.rstrip("s") not in category_stems]
    ok.sort(key=lambda x: (-x[1], x[0]))
    return [tok for tok, _ in ok]


def make_requirements(vendors, rng):
    categories = sorted(set().union(*vendors.category_set))
    extra = rng.choice(categories, N_REQUIREMENTS - len(categories), replace=False)
    needed = sorted(categories + [str(c) for c in extra])
    rng.shuffle(needed)
    reqs = []
    for i, cat in enumerate(needed, start=1):
        anchor = vendors.iloc[int(rng.integers(len(vendors)))]
        keywords = category_keywords(vendors, cat)
        keyword = (str(rng.choice(keywords))
                   if keywords and rng.random() < KEYWORD_PROB else "")
        reqs.append({"requirement_id": f"R{i:02d}", "needed_category": cat,
                     "buyer_lat": anchor.lat, "buyer_lon": anchor.lon,
                     "buyer_pincode": anchor.pincode,
                     "max_distance_km": int(rng.choice(RADII_KM)),
                     "preference_keyword": keyword,
                     "buyer_anchor_vendor_id": anchor.vendor_id})
    return pd.DataFrame(reqs)


def pick_candidates(req, vendors, rng):
    v = vendors[vendors.vendor_id != req.buyer_anchor_vendor_id].copy()
    v["distance_km"] = haversine_km(req.buyer_lat, req.buyer_lon, v.lat.values, v.lon.values)
    v["in_cat"] = v.category_set.map(lambda s: req.needed_category in s)
    v = v.sort_values(["distance_km", "vendor_id"])
    chosen = []

    def take(pool, n, stratum, random_pick):
        pool = pool[~pool.vendor_id.isin([c[0] for c in chosen])]
        if len(pool) < n:
            raise ValueError(f"{req.requirement_id}: only {len(pool)} vendors for {stratum}")
        ids = (pool.vendor_id.iloc[rng.choice(len(pool), n, replace=False)]
               if random_pick else pool.vendor_id.iloc[:n])
        chosen.extend((vid, stratum) for vid in ids)

    take(v[v.in_cat], 5, "near_in_category", random_pick=False)
    take(v[v.in_cat & (v.distance_km > req.max_distance_km)], 4, "far_in_category", True)
    take(v[~v.in_cat], 3, "near_other_category", random_pick=False)
    take(v, 3, "random", random_pick=True)
    assert [n for _, n in STRATA] == [5, 4, 3, 3]
    out = pd.DataFrame(chosen, columns=["vendor_id", "stratum"])
    return out.merge(v[["vendor_id", "distance_km"]], on="vendor_id")


def check_existing_sheet(sheet):
    """Compare an existing (possibly scored) sheet with the regenerated one, ignoring the
    rater/notes columns and the appended AI-judged expected_score column. Leaves the file
    untouched; stops if anything else differs."""
    rater_cols = ["rater_1_score", "rater_2_score", "notes"]
    fixed = [c for c in sheet.columns if c not in rater_cols]
    new = pd.read_csv(io.StringIO(sheet[fixed].to_csv(index=False, lineterminator="\n")),
                      dtype=str, keep_default_na=False)
    old = pd.read_csv(OUT_SHEET, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    allowed = [list(sheet.columns), list(sheet.columns) + ["expected_score"]]
    if list(old.columns) not in allowed or not old[fixed].equals(new):
        raise SystemExit(f"{OUT_SHEET} exists but differs from the regenerated pack outside the "
                         "rater columns - refusing to overwrite. Investigate before regenerating.")
    filled = int((old[["rater_1_score", "rater_2_score"]] != "").sum().sum())
    print(f"{OUT_SHEET} exists and matches the regenerated pack; left untouched "
          f"({filled} rater cells filled)")


def main():
    rng = np.random.default_rng(SEED)
    vendors = pd.read_csv(VENDORS, encoding="utf-8", dtype={"pincode": str},
                          keep_default_na=False, na_values={"locality": [""]})
    vendors["category_set"] = vendors.categories.str.split(";").map(set)

    reqs = make_requirements(vendors, rng)

    pairs = []
    for req in reqs.itertuples(index=False):
        c = pick_candidates(req, vendors, rng)
        c = c.iloc[rng.permutation(len(c))]  # hide stratum order from raters
        c.insert(0, "requirement_id", req.requirement_id)
        pairs.append(c)
    pairs = pd.concat(pairs, ignore_index=True)
    pairs.insert(0, "pair_id", [f"P{i:03d}" for i in range(1, len(pairs) + 1)])

    # ---- pair features (shared with the Phase 2 app: src/matching/features.py) ----
    p = pairs.merge(reqs, on="requirement_id").merge(vendors, on="vendor_id")
    for col, values in pair_features(p).items():
        p[col] = values
    pair_table = p[["pair_id", "requirement_id", "vendor_id", "stratum"] + FEATURES]

    # ---- checks ----
    assert len(reqs) == N_REQUIREMENTS and reqs.requirement_id.is_unique
    assert len(pair_table) == N_REQUIREMENTS * 15 and pair_table.pair_id.is_unique
    assert not pair_table.duplicated(["requirement_id", "vendor_id"]).any()
    assert (pair_table.groupby("requirement_id").size() == 15).all()
    assert not (p.vendor_id == p.buyer_anchor_vendor_id).any()
    stratum_counts = pair_table.groupby(["requirement_id", "stratum"]).size().unstack()
    assert (stratum_counts[[s for s, _ in STRATA]].values == [n for _, n in STRATA]).all()
    assert pair_table.loc[pair_table.stratum == "far_in_category", "within_radius"].eq(False).all()
    assert pair_table.loc[pair_table.stratum.str.contains("in_category"), "category_match"].all()
    assert not pair_table.loc[pair_table.stratum == "near_other_category", "category_match"].any()

    sheet = pd.DataFrame({
        "pair_id": p.pair_id, "requirement_id": p.requirement_id,
        "needed_category": p.needed_category, "buyer_lat": p.buyer_lat,
        "buyer_lon": p.buyer_lon, "buyer_pincode": p.buyer_pincode,
        "max_distance_km": p.max_distance_km, "preference_keyword": p.preference_keyword,
        "vendor_id": p.vendor_id, "vendor_title": p.title, "vendor_address": p.address,
        "vendor_categories": p.categories, "distance_km": p.distance_km.round(3),
        "rater_1_score": "", "rater_2_score": "", "notes": ""})

    OUT_REQ.parent.mkdir(parents=True, exist_ok=True)
    OUT_SHEET.parent.mkdir(parents=True, exist_ok=True)
    reqs.to_csv(OUT_REQ, index=False, encoding="utf-8", lineterminator="\n")
    pair_table.to_csv(OUT_PAIRS, index=False, encoding="utf-8", lineterminator="\n")
    if OUT_SHEET.exists():
        # never overwrite rater work: only confirm the existing sheet is the same pack
        check_existing_sheet(sheet)
    else:
        # utf-8-sig (UTF-8 with BOM) so Excel displays Tamil text correctly
        sheet.to_csv(OUT_SHEET, index=False, encoding="utf-8-sig", lineterminator="\n")
    OUT_RUBRIC.write_text(RUBRIC, encoding="utf-8")

    # ---- summary ----
    print(f"requirements: {len(reqs)} | pairs: {len(pair_table)}")
    print("requirements per category:", reqs.needed_category.value_counts().sort_index().to_dict())
    print("max_distance_km counts:", reqs.max_distance_km.value_counts().sort_index().to_dict())
    print(f"with preference keyword: {(reqs.preference_keyword != '').sum()} ->",
          reqs.loc[reqs.preference_keyword != "", ["needed_category", "preference_keyword"]]
          .apply(tuple, axis=1).tolist())
    print("pairs per stratum:", pair_table.stratum.value_counts().to_dict())
    print(pair_table.groupby("stratum")[["within_radius", "category_match", "keyword_in_title"]]
          .mean().round(3).to_string())
    print("overall feature means:",
          pair_table[["within_radius", "category_match", "keyword_in_title", "is_chain"]]
          .mean().round(3).to_dict())
    print(f"distance_km: min {pair_table.distance_km.min():.3f}, "
          f"median {pair_table.distance_km.median():.3f}, max {pair_table.distance_km.max():.3f}")
    print(f"wrote {OUT_REQ}, {OUT_PAIRS}, {OUT_RUBRIC} (annotation sheet: {OUT_SHEET})")


if __name__ == "__main__":
    main()
