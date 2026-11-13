"""Matching pipeline: requirement -> candidates -> features -> rules -> score -> rank -> explain.

Candidates: every vendor within max(2 x max_distance_km, 5 km), capped at the 200 nearest.
If fewer than 10 of those are eligible, every R1-category vendor outside that set is added.
For an existing requirement, its buyer-anchor vendor (whose location was copied as the
buyer location) is excluded, as in Phase 1.
"""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from matching import eligibility as E
from matching.artifacts import load_classifier, load_linear_model, pharmacy_probability
from matching.explain import PHARMACY, explain
from matching.features import FEATURES, category_set, clean_keyword, pair_features
from matching.ranking import CHAIN_PREFERENCES, rank
from matching.scoring import contributions, display_score, predict_raw

ROOT = Path(__file__).resolve().parents[2]
VENDORS = ROOT / "data/processed/vendors.csv"
REQUIREMENTS = ROOT / "data/processed/requirements.csv"
MIN_RANGE_KM = 5.0
RANGE_FACTOR = 2.0
MAX_NEARBY = 200
MIN_ELIGIBLE = 10


# ---------------------------------------------------------------- data
def load_vendors(path=VENDORS):
    return pd.read_csv(path, encoding="utf-8", dtype={"pincode": str},
                       keep_default_na=False, na_values={"locality": [""]})


def load_requirements(path=REQUIREMENTS):
    return pd.read_csv(path, encoding="utf-8", dtype={"buyer_pincode": str},
                       keep_default_na=False)


def all_categories(vendors):
    return sorted(set().union(*vendors.categories.map(category_set)))


def make_requirement(needed_category, buyer_lat, buyer_lon, max_distance_km,
                     preference_keyword="", keyword_required=False, chain_preference="none",
                     requirement_id="NEW", buyer_anchor_vendor_id=None):
    if float(max_distance_km) <= 0:
        raise ValueError("max_distance_km must be > 0")
    if not (-90 <= float(buyer_lat) <= 90 and -180 <= float(buyer_lon) <= 180):
        raise ValueError("buyer location out of range")
    if chain_preference not in CHAIN_PREFERENCES:
        raise ValueError(f"chain_preference must be one of {CHAIN_PREFERENCES}")
    return {"requirement_id": requirement_id, "needed_category": needed_category,
            "buyer_lat": float(buyer_lat), "buyer_lon": float(buyer_lon),
            "max_distance_km": float(max_distance_km),
            "preference_keyword": clean_keyword(preference_keyword),
            "keyword_required": bool(keyword_required), "chain_preference": chain_preference,
            "buyer_anchor_vendor_id": buyer_anchor_vendor_id}


def requirement_from_id(requirements, requirement_id, keyword_required=False,
                        chain_preference="none"):
    r = requirements.set_index("requirement_id").loc[requirement_id]
    return make_requirement(r.needed_category, r.buyer_lat, r.buyer_lon, r.max_distance_km,
                            r.preference_keyword, keyword_required, chain_preference,
                            requirement_id, r.buyer_anchor_vendor_id)


# ---------------------------------------------------------------- steps
def pair_frame(requirement, vendors):
    """Vendors joined with the requirement plus the shared pair features."""
    p = vendors.copy()
    for k in ["needed_category", "buyer_lat", "buyer_lon", "max_distance_km",
              "preference_keyword"]:
        p[k] = requirement[k]
    feats = pair_features(p)
    return pd.concat([vendors, feats.drop(columns=["n_categories", "is_chain"])], axis=1)


def select_candidates(requirement, vendors):
    v = vendors
    if requirement.get("buyer_anchor_vendor_id"):
        v = v[v.vendor_id != requirement["buyer_anchor_vendor_id"]]
    p = pair_frame(requirement, v.reset_index(drop=True))
    p = p.sort_values(["distance_km", "vendor_id"], kind="mergesort").reset_index(drop=True)
    radius = max(RANGE_FACTOR * requirement["max_distance_km"], MIN_RANGE_KM)
    nearby = p[p.distance_km <= radius].head(MAX_NEARBY).assign(candidate_source="nearby")
    rules = [E.check_rules(requirement, r, r["distance_km"]) for _, r in nearby.iterrows()]
    n_eligible = sum(E.is_eligible(x) for x in rules)
    meta = {"search_radius_km": radius, "n_nearby": len(nearby),
            "n_eligible_nearby": int(n_eligible), "extended": False, "n_extension": 0}
    if n_eligible < MIN_ELIGIBLE:
        ext = p[~p.vendor_id.isin(nearby.vendor_id) & p.category_match]
        meta.update(extended=True, n_extension=len(ext))
        nearby = pd.concat([nearby, ext.assign(candidate_source="category_extension")])
    return nearby.reset_index(drop=True), meta


@dataclass
class MatchResult:
    requirement: dict
    eligible: pd.DataFrame
    ineligible: pd.DataFrame
    explanations: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def run_match(requirement, vendors, model, classifier=None):
    cands, meta = select_candidates(requirement, vendors)
    return score_and_rank(requirement, cands, model, classifier, meta)


def score_and_rank(requirement, cands, model, classifier=None, meta=None):
    """Rules, score, explanations and ranking for a given candidate frame (pair_frame rows).
    Used by run_match and, on the annotated pairs, by the Phase 2 evaluation."""
    cands = cands.reset_index(drop=True).copy()
    meta = dict(meta or {})
    rules = [E.check_rules(requirement, r, r["distance_km"]) for _, r in cands.iterrows()]
    cands["eligible"] = [E.is_eligible(x) for x in rules]
    cands["n_failed"] = [len(E.failed_rules(x)) for x in rules]
    cands["failed_rules"] = ["; ".join(f"{f['rule']}: {f['detail']}" for f in E.failed_rules(x))
                             for x in rules]
    cands["raw_score"] = predict_raw(model, cands)
    cands["display_score"] = display_score(cands.raw_score.to_numpy())
    contrib = contributions(model, cands)
    probs = None
    if requirement["needed_category"] == PHARMACY and classifier is not None:
        probs = pharmacy_probability(classifier, cands)
        cands["pharmacy_probability"] = probs
    explanations = {
        r["vendor_id"]: explain(r, rules[i], contrib.loc[i], model["beta"],
                                None if probs is None else probs[i])
        for i, r in cands.iterrows()}
    eligible, ineligible = rank(cands, requirement["chain_preference"])
    meta.update(n_candidates=len(cands), n_eligible=len(eligible),
                n_ineligible=len(ineligible))
    return MatchResult(requirement, eligible, ineligible, explanations, meta)


class Matcher:
    """Loads data and artifacts once (from data/processed and artifacts/ only)."""

    def __init__(self):
        self.vendors = load_vendors()
        self.requirements = load_requirements()
        self.categories = all_categories(self.vendors)
        self.model = load_linear_model()
        self.classifier = load_classifier()

    def match(self, requirement):
        if requirement["needed_category"] not in self.categories:
            raise ValueError(f"unknown category {requirement['needed_category']!r}")
        return run_match(requirement, self.vendors, self.model, self.classifier)

    def match_id(self, requirement_id, keyword_required=False, chain_preference="none"):
        return self.match(requirement_from_id(self.requirements, requirement_id,
                                              keyword_required, chain_preference))


RESULT_COLUMNS = ["rank", "vendor_id", "title", "distance_km", "raw_score", "display_score",
                  "eligible"] + FEATURES
