# 2. The features (what goes into X)

## A. Regression features (one row per requirement–vendor pair, 450 rows)

| Feature | Type | How it is computed | Why it is there |
|---|---|---|---|
| `distance_km` | continuous | Haversine (great-circle) distance between the buyer's lat/lon and the vendor's lat/lon | Closer vendors are better |
| `within_radius` | 0/1 | 1 if `distance_km ≤ max_distance_km` of the requirement | The buyer's hard distance limit |
| `category_match` | 0/1 | 1 if the needed category is among the vendor's listed categories | The most important criterion: is it the right kind of shop? |
| `keyword_in_title` | 0/1 | 1 if the requirement's preference keyword (e.g. "eyeglasses") is a word in the vendor title | The buyer's optional preference |
| `n_categories` | count | Number of category files the vendor appears in (1 to 10) | Vendors listed everywhere are often generic or mislabelled |
| `is_chain` | 0/1 | 1 if the cleaned title is shared by ≥ 3 vendors | Chains may be more reliable (limitation: generic names like "ration shop" also count) |
| intercept | constant 1 | | Lets the model have a baseline β₀ |

So X has **p = 7 columns** (intercept + 6). The target y is the AI-judged
`expected_score` (0–100).

**Feature scale matters when reading β:** `distance_km` is per km, the 0/1 features are "on vs
off", `n_categories` is per extra category.

## B. Classification features (one row per vendor, 4,612 rows)

| Block | Count | How |
|---|---|---|
| Title-token indicators | 421 | 1 if the word appears in the vendor's cleaned title. Vocabulary = words used by ≥ 5 **training** vendors (`min_df = 5`) |
| Pincode indicators | 20 | 1 if the vendor's pincode is one of the 20 most common **training** pincodes; every other pincode is the all-zeros reference level ("other") |
| Intercept | 1 | |
| **Total** | **442** | after dropping 6 token columns that were exact linear combinations of earlier ones (`get`, `pazhamudir`, `price`, `replacement`, `specsmakers`, `up`) |

**Why drop dependent columns?** If one column equals a combination of others (e.g. *Kovai
Pazhamudir Nilayam* always appear together), XᵀX is singular and β is not unique. After
dropping them, X_train has full rank (442 of 442).

**Why a reference level for pincodes?** With an intercept, a full one-hot block sums to the
intercept column (the "dummy variable trap"). Leaving "other" as all-zeros avoids that.

**Why "fitted on train only"?** If the vocabulary used test vendors, information from the
test set would leak into training and inflate test scores.

## C. What is NOT a feature
`count` and `outside` in the raw file headers (meaning unknown), and anything from the test
set. The classification label (`is_pharmacy`) comes from which file a vendor appears in, so
it is a **weak label**: some "pharmacies" are a chemistry academy or a supermarket.
