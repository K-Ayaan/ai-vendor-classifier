"""Deterministic eligibility rules. No model is involved: a vendor is eligible only if every
mandatory rule passes, and the ranking never puts an ineligible vendor above an eligible one.

R1 category  the vendor is listed in needed_category (its source-file categories)
R2 distance  distance_km <= max_distance_km  (distance_km as computed by features.py)
R3 keyword   only when keyword_required: the keyword is a token of the vendor title;
             otherwise the rule is skipped (passed=True, skipped=True)

Each rule returns {"rule", "passed", "skipped", "detail"} with the actual values in detail.
"""
from matching.features import category_set, clean_keyword, keyword_in_title

R1, R2, R3 = "R1 category", "R2 distance", "R3 keyword"


def _result(rule, passed, detail, skipped=False):
    return {"rule": rule, "passed": bool(passed), "skipped": skipped, "detail": detail}


def rule_category(needed_category, vendor_categories):
    cats = sorted(category_set(vendor_categories))
    if needed_category in cats:
        return _result(R1, True, f"listed in {needed_category}")
    return _result(R1, False, f"not listed in {needed_category} (listed in: {', '.join(cats)})")


def rule_distance(distance_km, max_distance_km):
    d, limit = float(distance_km), float(max_distance_km)
    if d <= limit:
        return _result(R2, True, f"{d:.2f} km <= {limit:g} km limit")
    return _result(R2, False, f"{d:.2f} km > {limit:g} km limit")


def rule_keyword(keyword, title, keyword_required):
    kw = clean_keyword(keyword)
    if not keyword_required:
        return _result(R3, True, "skipped: keyword not required", skipped=True)
    if not kw:
        return _result(R3, True, "skipped: no keyword given", skipped=True)
    if keyword_in_title(kw, title):
        return _result(R3, True, f"'{kw}' in title")
    return _result(R3, False, f"'{kw}' not in title '{title}'")


def check_rules(requirement, vendor, distance_km):
    """All mandatory rules for one vendor. `requirement` needs needed_category,
    max_distance_km, preference_keyword, keyword_required; `vendor` needs categories, title."""
    return [rule_category(requirement["needed_category"], vendor["categories"]),
            rule_distance(distance_km, requirement["max_distance_km"]),
            rule_keyword(requirement.get("preference_keyword", ""), vendor["title"],
                         bool(requirement.get("keyword_required", False)))]


def is_eligible(results):
    return all(r["passed"] for r in results)


def failed_rules(results):
    return [r for r in results if not r["passed"]]
