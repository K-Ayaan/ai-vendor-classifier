"""Per-vendor explanations built only from computed values: rule results with the actual
numbers, feature values, coefficient x value contributions, and (pharmacy requirements only)
the L2 classifier's probability as an advisory category check."""
from matching.features import FEATURES

PHARMACY = "pharmacy"
CATEGORY_CHECK_THRESHOLD = 0.5


def category_check(probability):
    """Advisory only: it never changes eligibility or rank."""
    if probability is None:
        return None
    warn = probability < CATEGORY_CHECK_THRESHOLD
    return {"pharmacy_probability": float(probability), "warning": bool(warn),
            "message": (f"Category check: the pharmacy classifier gives {probability:.2f} "
                        f"(< {CATEGORY_CHECK_THRESHOLD}); this vendor may not be a pharmacy "
                        "despite its listing." if warn else
                        f"Category check: pharmacy classifier gives {probability:.2f}.")}


def explain(row, rules, contrib_row, beta, pharmacy_probability=None):
    terms = [{"term": "intercept", "coefficient": float(beta[0]), "value": 1.0,
              "contribution": float(contrib_row["intercept"])}]
    for j, f in enumerate(FEATURES, start=1):
        terms.append({"term": f, "coefficient": float(beta[j]), "value": float(row[f]),
                      "contribution": float(contrib_row[f])})
    return {"vendor_id": row["vendor_id"], "title": row["title"],
            "eligible": bool(row["eligible"]), "rules": rules,
            "features": {f: (row[f].item() if hasattr(row[f], "item") else row[f])
                         for f in FEATURES},
            "contributions": terms, "raw_score": float(row["raw_score"]),
            "display_score": float(row["display_score"]),
            "category_check": category_check(pharmacy_probability)}
