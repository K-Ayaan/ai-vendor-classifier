"""The one feature module used for training (Phase 1 scripts) and inference (Phase 2 app).

Pair features (requirement x vendor), used by the linear match-score model:
    distance_km       straight-line (haversine) distance buyer -> vendor, rounded to 4 dp
    within_radius     distance_km <= max_distance_km
    category_match    needed_category is one of the vendor's listed categories
    keyword_in_title  the preference keyword is a token of the normalised vendor title
    n_categories      number of source files the vendor appears in
    is_chain          normalised title shared by >= 3 vendors (see preprocessing.py)

Vendor features, used by the is_pharmacy classifier: FeatureBuilder (title tokens +
top-k pincode one-hot), fitted on the training split only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import haversine_km, norm_title  # noqa: E402

FEATURES = ["distance_km", "within_radius", "category_match", "keyword_in_title",
            "n_categories", "is_chain"]
DISTANCE_DECIMALS = 4
# columns pair_features() reads
REQUIREMENT_COLUMNS = ["needed_category", "buyer_lat", "buyer_lon", "max_distance_km",
                       "preference_keyword"]
VENDOR_COLUMNS = ["title", "lat", "lon", "categories", "n_categories", "is_chain"]

MIN_DF = 5            # classifier: a title token must appear in >= 5 training titles
TOP_K_PINCODES = 20   # classifier: one-hot for the 20 most frequent training pincodes


# ---------------------------------------------------------------- pair features
def title_token_list(title):
    return norm_title(title).split()


def title_tokens(title):
    return set(title_token_list(title))


def category_set(categories):
    """'a;b' -> {'a', 'b'} (vendors.csv stores the multi-label list ';'-separated)."""
    return set(categories.split(";")) if isinstance(categories, str) else set(categories)


def clean_keyword(keyword):
    """'' for a missing keyword; otherwise normalised like a title."""
    if keyword is None or (isinstance(keyword, float) and np.isnan(keyword)):
        return ""
    return norm_title(str(keyword))


def keyword_in_title(keyword, title):
    kw = clean_keyword(keyword)
    return bool(kw) and set(kw.split()) <= title_tokens(title)


def pair_features(p):
    """Feature frame (columns FEATURES, same index as p) for a frame with one row per
    requirement-vendor pair holding REQUIREMENT_COLUMNS and VENDOR_COLUMNS."""
    missing = [c for c in REQUIREMENT_COLUMNS + VENDOR_COLUMNS if c not in p]
    if missing:
        raise KeyError(f"pair_features: missing columns {missing}")
    dist = pd.Series(haversine_km(p.buyer_lat.to_numpy(float), p.buyer_lon.to_numpy(float),
                                  p.lat.to_numpy(float), p.lon.to_numpy(float)),
                     index=p.index).round(DISTANCE_DECIMALS)
    out = pd.DataFrame(index=p.index)
    out["distance_km"] = dist
    out["within_radius"] = dist <= p.max_distance_km
    out["category_match"] = [r.needed_category in category_set(r.categories)
                             for r in p.itertuples()]
    out["keyword_in_title"] = [keyword_in_title(r.preference_keyword, r.title)
                               for r in p.itertuples()]
    out["n_categories"] = p.n_categories.astype(int)
    out["is_chain"] = p.is_chain.astype(bool)
    return out[FEATURES]


def design_matrix(df):
    """[1, FEATURES] as floats - the linear model's design matrix."""
    return np.column_stack([np.ones(len(df))] + [df[f].astype(float).to_numpy() for f in FEATURES])


# ---------------------------------------------------------------- classifier features
def independent_columns(X, tol=1e-8):
    """Indices of columns kept by a left-to-right Gram-Schmidt pass: a column is dropped
    when it is (numerically) a linear combination of the columns kept before it."""
    basis, keep = [], []
    for j in range(X.shape[1]):
        v = X[:, j].astype(float)
        norm0 = np.linalg.norm(v)
        for _ in range(2):  # re-orthogonalise for numerical stability
            for q in basis:
                v = v - (q @ v) * q
        if norm0 > 0 and np.linalg.norm(v) > tol * norm0:
            basis.append(v / np.linalg.norm(v))
            keep.append(j)
    return np.array(keep)


class FeatureBuilder:
    """Title-token indicators + top-k pincode one-hot. Fitted on the training split only.

    Pincodes outside the train top-k form the reference level "other" (all zeros), so
    the one-hot block is not collinear with the intercept. Columns that are linear
    combinations of earlier columns in train (exact duplicates such as "hap"/"daily", and
    shop-name tokens that always co-occur) are dropped, so X_train has full column rank
    and beta is identifiable.
    """

    def fit(self, train):
        self.vec = CountVectorizer(tokenizer=title_token_list, lowercase=False,
                                   token_pattern=None, binary=True, min_df=MIN_DF)
        self.vec.fit(train.title)
        vc = train.pincode.value_counts()
        self.pincodes = sorted(vc.index, key=lambda p: (-vc[p], p))[:TOP_K_PINCODES]
        all_names = (["intercept"] + [f"tok={t}" for t in self.vec.get_feature_names_out()]
                     + [f"pin={p}" for p in self.pincodes])
        self.keep = independent_columns(self._candidates(train))
        self.names = [all_names[i] for i in self.keep]
        self.dropped_dependent = sorted(set(all_names) - set(self.names))
        return self

    def _candidates(self, df):
        tokens = self.vec.transform(df.title).toarray()
        pins = np.column_stack([(df.pincode == p).to_numpy() for p in self.pincodes])
        return np.column_stack([np.ones(len(df)), tokens, pins]).astype(float)

    def transform(self, df):
        return self._candidates(df)[:, self.keep]
