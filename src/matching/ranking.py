"""Ranking. Eligibility is decided before ranking and is never overridden by the score:
every eligible vendor is placed above every ineligible one.

Eligible vendors: chain-preference group (if any), then raw predicted score (descending),
then distance (ascending), then vendor_id (makes the order fully deterministic).
Ineligible vendors (listed separately): fewest failed rules, then the same keys.
"""
import numpy as np

CHAIN_PREFERENCES = ("none", "exclude", "prefer")


def _chain_group(df, chain_preference):
    if chain_preference not in CHAIN_PREFERENCES:
        raise ValueError(f"chain_preference must be one of {CHAIN_PREFERENCES}")
    if chain_preference == "exclude":   # chains go below non-chains (still eligible)
        return df.is_chain.astype(int)
    if chain_preference == "prefer":    # chains go above non-chains
        return 1 - df.is_chain.astype(int)
    return np.zeros(len(df), dtype=int)


def rank(candidates, chain_preference="none"):
    """candidates needs: vendor_id, eligible, n_failed, raw_score, distance_km, is_chain.
    Returns (eligible, ineligible), each sorted; eligible gets rank 1..n."""
    df = candidates.assign(_chain=_chain_group(candidates, chain_preference),
                           _neg=-candidates.raw_score)
    keys = ["_chain", "_neg", "distance_km", "vendor_id"]
    eligible = df[df.eligible].sort_values(keys, kind="mergesort")
    ineligible = df[~df.eligible].sort_values(["n_failed"] + keys, kind="mergesort")
    eligible = eligible.drop(columns=["_chain", "_neg"]).reset_index(drop=True)
    ineligible = ineligible.drop(columns=["_chain", "_neg"]).reset_index(drop=True)
    eligible.insert(0, "rank", np.arange(1, len(eligible) + 1))
    return eligible, ineligible
