import pandas as pd
import pytest

from matching.eligibility import (R1, R2, R3, check_rules, failed_rules, is_eligible,
                                  rule_category, rule_distance, rule_keyword)

REQ = {"needed_category": "pharmacy", "max_distance_km": 3, "preference_keyword": "medicals",
       "keyword_required": True}


# ---- R1 category
def test_r1_passes_when_listed():
    r = rule_category("pharmacy", "grocery_supermarket;pharmacy")
    assert r == {"rule": R1, "passed": True, "skipped": False, "detail": "listed in pharmacy"}


def test_r1_fails_when_not_listed():
    r = rule_category("pharmacy", "fruits;vegetables")
    assert not r["passed"]
    assert r["detail"] == "not listed in pharmacy (listed in: fruits, vegetables)"


def test_r1_is_exact_not_substring():
    assert not rule_category("fruits", "fruits_juice")["passed"]


# ---- R2 distance
def test_r2_fails_beyond_limit():
    r = rule_distance(4.2, 3)
    assert not r["passed"] and r["detail"] == "4.20 km > 3 km limit"


def test_r2_passes_within_and_at_limit():
    assert rule_distance(1.5, 3)["passed"]
    assert rule_distance(3.0, 3)["passed"]  # <= is inclusive, like within_radius


# ---- R3 keyword
def test_r3_passes_when_required_and_in_title():
    r = rule_keyword("medicals", "Sri Balaji Medicals", keyword_required=True)
    assert r["passed"] and not r["skipped"]


def test_r3_fails_when_required_and_missing():
    r = rule_keyword("medicals", "Apollo Pharmacy", keyword_required=True)
    assert not r["passed"] and "not in title" in r["detail"]


def test_r3_matches_whole_tokens_only():
    assert not rule_keyword("fresh", "Freshmart", keyword_required=True)["passed"]


def test_r3_skipped_when_not_required():
    r = rule_keyword("medicals", "Apollo Pharmacy", keyword_required=False)
    assert r["passed"] and r["skipped"] and r["detail"] == "skipped: keyword not required"


def test_r3_skipped_when_required_but_no_keyword():
    for kw in ["", None, float("nan")]:
        r = rule_keyword(kw, "Apollo Pharmacy", keyword_required=True)
        assert r["passed"] and r["skipped"]


# ---- combined
def test_check_rules_all_pass():
    v = {"categories": "pharmacy", "title": "Sri Balaji Medicals"}
    res = check_rules(REQ, v, 2.0)
    assert [r["rule"] for r in res] == [R1, R2, R3] and is_eligible(res)


def test_check_rules_reports_every_failure():
    v = {"categories": "fruits", "title": "Handicrafts Emporium"}
    res = check_rules(REQ, v, 4.2)
    assert not is_eligible(res)
    assert [r["rule"] for r in failed_rules(res)] == [R1, R2, R3]


@pytest.fixture(scope="module")
def sheet_pairs():
    pairs = pd.read_csv("data/processed/pairs.csv")
    sheet = pd.read_csv("data/annotation/annotation_sheet.csv", encoding="utf-8-sig",
                        dtype={"buyer_pincode": str}, keep_default_na=False)
    return pairs.merge(sheet[["pair_id", "max_distance_km", "needed_category",
                              "vendor_categories"]], on="pair_id")


def test_r1_r2_agree_with_shared_features_on_all_450_pairs(sheet_pairs):
    """The rules and the model features use one definition of category and distance."""
    assert len(sheet_pairs) == 450
    for r in sheet_pairs.itertuples():
        assert rule_category(r.needed_category, r.vendor_categories)["passed"] == r.category_match
        assert rule_distance(r.distance_km, r.max_distance_km)["passed"] == r.within_radius
