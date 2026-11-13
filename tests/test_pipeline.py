import numpy as np
import pandas as pd
import pytest

from matching.pipeline import (Matcher, make_requirement, pair_frame, requirement_from_id,
                               run_match)
from matching.features import FEATURES
from matching.ranking import rank
from matching.scoring import SCORE_LABEL, contributions, predict_raw


@pytest.fixture(scope="module")
def matcher():
    return Matcher()


@pytest.fixture(scope="module")
def sheet():
    s = pd.read_csv("data/annotation/annotation_sheet.csv", encoding="utf-8-sig",
                    dtype={"buyer_pincode": str}, keep_default_na=False)
    pairs = pd.read_csv("data/processed/pairs.csv")
    return s.merge(pairs, on=["pair_id", "requirement_id", "vendor_id"],
                   suffixes=("_sheet", ""), validate="one_to_one")


# ---------------------------------------------------------------- eligibility before score
def test_ineligible_with_higher_score_never_outranks_eligible():
    df = pd.DataFrame({"vendor_id": ["A", "B", "C", "D"],
                       "eligible": [False, True, False, True], "n_failed": [1, 0, 2, 0],
                       "raw_score": [99.0, 10.0, 150.0, -5.0],
                       "distance_km": [0.1, 2.0, 0.2, 1.0], "is_chain": [False] * 4})
    for pref in ["none", "exclude", "prefer"]:
        el, inel = rank(df, pref)
        order = list(el.vendor_id) + list(inel.vendor_id)
        assert order[:2] == ["B", "D"] and set(order[2:]) == {"A", "C"}


def test_ranking_invariant_on_real_requirements(matcher):
    for rid in matcher.requirements.requirement_id:
        res = matcher.match_id(rid)
        assert res.eligible.eligible.all() and not res.ineligible.eligible.any()
        assert list(res.eligible["rank"]) == list(range(1, len(res.eligible) + 1))
        # sorted by raw score, ties by distance
        s = res.eligible[["raw_score", "distance_km"]].to_numpy()
        for a, b in zip(s[:-1], s[1:]):
            assert a[0] > b[0] or (a[0] == b[0] and a[1] <= b[1])


def test_chain_preference_reorders_but_never_changes_eligibility(matcher):
    base = matcher.match_id("R01")
    for pref in ["exclude", "prefer"]:
        res = matcher.match_id("R01", chain_preference=pref)
        assert set(res.eligible.vendor_id) == set(base.eligible.vendor_id)
        chains = res.eligible.is_chain.to_numpy()
        expected = np.sort(chains) if pref == "exclude" else np.sort(chains)[::-1]
        assert (chains == expected).all()


# ---------------------------------------------------------------- explanations
def test_contributions_sum_to_raw_prediction(matcher):
    total, worst = 0, 0.0
    for rid in matcher.requirements.requirement_id:
        res = matcher.match_id(rid)
        for vid, e in res.explanations.items():
            s = sum(t["contribution"] for t in e["contributions"])
            worst = max(worst, abs(s - e["raw_score"]))
            total += 1
    assert total > 0 and worst <= 1e-9


def test_display_score_is_clipped_raw(matcher):
    res = matcher.match_id("R01")
    allr = pd.concat([res.eligible, res.ineligible])
    assert np.allclose(allr.display_score, allr.raw_score.clip(0, 100))
    assert SCORE_LABEL == "Predicted match score (model trained on AI-judged scores)"


def test_pharmacy_check_only_for_pharmacy_and_never_changes_eligibility(matcher):
    reqs = matcher.requirements
    for rid, cat in zip(reqs.requirement_id, reqs.needed_category):
        res = matcher.match_id(rid)
        checks = [e["category_check"] for e in res.explanations.values()]
        if cat == "pharmacy":
            assert all(c is not None for c in checks)
            no_clf = run_match(res.requirement, matcher.vendors, matcher.model, None)
            assert list(no_clf.eligible.vendor_id) == list(res.eligible.vendor_id)
        else:
            assert all(c is None for c in checks)


# ---------------------------------------------------------------- shared features
def test_pipeline_features_equal_annotation_features_for_all_450_pairs(matcher, sheet):
    reqs = matcher.requirements
    v = matcher.vendors.set_index("vendor_id")
    n = 0
    for rid, grp in sheet.groupby("requirement_id"):
        req = requirement_from_id(reqs, rid)
        p = pair_frame(req, v.loc[grp.vendor_id].reset_index()).set_index("vendor_id")
        g = grp.set_index("vendor_id")
        for f in FEATURES:
            assert (p.loc[g.index, f].to_numpy() == g[f].to_numpy()).all(), (rid, f)
        assert (p.loc[g.index, "distance_km"].round(3).to_numpy()
                == g.distance_km_sheet.to_numpy()).all()
        n += len(g)
    assert n == 450


def test_pipeline_scores_equal_phase1_predictions(matcher):
    """Saved model + pipeline features reproduce Phase 1's fitted values for all 450 pairs."""
    pred = pd.read_csv("data/processed/regression_predictions.csv")
    pairs = pd.read_csv("data/processed/pairs.csv").merge(pred, on=["pair_id", "requirement_id"])
    assert len(pairs) == 450
    assert np.max(np.abs(predict_raw(matcher.model, pairs) - pairs.fitted)) < 1e-9
    c = contributions(matcher.model, pairs)
    assert np.max(np.abs(c.sum(axis=1) - pairs.fitted)) < 1e-9


# ---------------------------------------------------------------- determinism & e2e
def test_same_input_same_ranking_twice(matcher):
    req = make_requirement("bakery_sweets", 12.85, 80.05, 3, "cake", True, "exclude")
    a, b = matcher.match(req), matcher.match(req)
    pd.testing.assert_frame_equal(a.eligible, b.eligible)
    pd.testing.assert_frame_equal(a.ineligible, b.ineligible)


@pytest.mark.parametrize("rid", ["R01", "R09", "R16"])
def test_end_to_end_existing_requirement(matcher, rid):
    res = matcher.match_id(rid)
    req = res.requirement
    assert res.meta["n_candidates"] == len(res.eligible) + len(res.ineligible) > 0
    assert req["buyer_anchor_vendor_id"] not in set(res.eligible.vendor_id) | set(
        res.ineligible.vendor_id)
    assert (res.eligible.category_match & (res.eligible.distance_km <= req["max_distance_km"])).all()
    assert (res.ineligible.failed_rules != "").all()
    assert set(res.explanations) == set(res.eligible.vendor_id) | set(res.ineligible.vendor_id)
    if res.meta["n_eligible_nearby"] >= 10:
        assert not res.meta["extended"] and res.meta["n_nearby"] <= 200
    else:
        assert res.meta["extended"]


def test_keyword_required_filters(matcher):
    req = requirement_from_id(matcher.requirements, "R01", keyword_required=True)
    res = matcher.match(req)
    if req["preference_keyword"]:
        assert res.eligible.keyword_in_title.all()


def test_new_requirement_validation():
    with pytest.raises(ValueError):
        make_requirement("pharmacy", 12.9, 80.1, 0)
    with pytest.raises(ValueError):
        make_requirement("pharmacy", 12.9, 80.1, 3, chain_preference="maybe")
