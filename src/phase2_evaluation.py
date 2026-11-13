"""Phase 2 Step 5 - evaluate the matching pipeline on the 6 held-out test requirements.

Usage: python src/phase2_evaluation.py
Writes reports/phase2_metrics.json and reports/phase2_results.md.

Only the 15 annotated pairs of each requirement have an AI-judged expected_score, so each
method re-ranks exactly those 15 vendors ("judged pool"). Relevant = expected_score >= 70.
Methods:
  pipeline   eligibility (R1 category, R2 distance; R3 off) then predicted score
             (the app's own code path: pipeline.score_and_rank)
  nearest    the same eligibility, then distance (nearest first)
  random     one random permutation per requirement, numpy default_rng(42)
Metrics: NDCG@5 and precision@5 (binary relevance), and the number of relevant vendors
that the eligibility rules exclude (ranked below every eligible vendor).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching.artifacts import load_classifier, load_linear_model, load_model_card  # noqa: E402
from matching.features import title_tokens  # noqa: E402
from matching.pipeline import (load_requirements, load_vendors, pair_frame,  # noqa: E402
                               requirement_from_id, score_and_rank)

SEED = 42
K = 5
RELEVANT_AT = 70
N_RANDOM_CONTEXT = 1000
SHEET = Path("data/annotation/annotation_sheet.csv")
RATIONALES = Path("data/annotation/ai_scores.csv")
PAIRS = Path("data/processed/pairs.csv")
OUT_JSON = Path("reports/phase2_metrics.json")
OUT_MD = Path("reports/phase2_results.md")
METHODS = ["pipeline", "nearest", "random"]


def ndcg_at_k(rel, k=K):
    rel = np.asarray(rel, float)
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(rel[:k] * disc[:len(rel[:k])]))
    ideal = np.sort(rel)[::-1][:k]
    idcg = float(np.sum(ideal * disc[:len(ideal)]))
    return dcg / idcg if idcg > 0 else float("nan")


def precision_at_k(rel, k=K):
    return float(np.sum(np.asarray(rel)[:k]) / k)


def evaluate_requirement(req, pool, model, classifier, rng):
    """pool: annotated vendors for one requirement with expected_score."""
    res = score_and_rank(req, pool, model, classifier)
    scored = pd.concat([res.eligible, res.ineligible], ignore_index=True)
    orders = {
        "pipeline": list(scored.vendor_id),
        "nearest": (list(res.eligible.sort_values(["distance_km", "vendor_id"]).vendor_id)
                    + list(res.ineligible.sort_values(["distance_km", "vendor_id"]).vendor_id)),
    }
    ids = sorted(pool.vendor_id)
    orders["random"] = [ids[i] for i in rng.permutation(len(ids))]
    score = pool.set_index("vendor_id").expected_score
    rel = (score >= RELEVANT_AT)
    out = {}
    for name, order in orders.items():
        r = rel.loc[order].to_numpy().astype(int)
        out[name] = {"ndcg@5": ndcg_at_k(r), "precision@5": precision_at_k(r),
                     "top5": order[:K]}
    excluded = scored[~scored.eligible & scored.vendor_id.map(rel)]
    elig = {"n_pairs": len(pool), "n_relevant": int(rel.sum()),
            "n_eligible": len(res.eligible),
            "relevant_excluded_by_eligibility": len(excluded),
            "relevant_excluded": [{"vendor_id": r.vendor_id, "title": r.title,
                                   "expected_score": int(score[r.vendor_id]),
                                   "failed_rules": r.failed_rules} for r in excluded.itertuples()],
            "eligible_not_relevant": int((scored.eligible & ~scored.vendor_id.map(rel)).sum())}
    return out, elig, scored.assign(expected_score=scored.vendor_id.map(score).to_numpy())


def error_reason(r, beta):
    """Reason for a large score error, built from the pair's actual feature values and
    coefficient x value contributions."""
    under = r.display_score < r.expected_score
    reasons = []
    if under and r.n_categories >= 5:
        reasons.append(f"listed in {r.n_categories} categories, so the n_categories term adds "
                       f"{beta['n_categories'] * r.n_categories:.2f}; the judge saw a large "
                       "multi-category store as a strong fit")
    if not under and r.category_match and r.expected_score <= 30:
        reasons.append(f"listed in {r.needed_category} (category_match adds "
                       f"{beta['category_match']:.2f}) but the judge read the title as a "
                       "different or weaker kind of shop: noisy source-file category label")
    if under and not r.category_match:
        reasons.append(f"not listed in {r.needed_category}, so no category credit"
                       + ("" if r.within_radius else
                          f", and {r.distance_km:.2f} km is outside the "
                          f"{r.max_distance_km:g} km limit")
                       + "; the judge still gave partial credit")
    kw = r.preference_keyword
    if under and kw and not r.keyword_in_title:
        stems = {t.rstrip("s") for t in title_tokens(r.title)}
        if kw.rstrip("s") in stems:
            reasons.append(f"keyword '{kw}' is not a whole token of the title "
                           f"('{r.title}'), so keyword_in_title = False")
    if not reasons:
        reasons.append("model over-scores: the features cannot see what the judge penalised"
                       if not under else
                       "model under-scores: the judge credited something the features do not "
                       "encode")
    return "; ".join(reasons)


def main():
    card = load_model_card()
    test_ids = card["linear_model"]["training_data"]["test_requirements"]
    model, clf = load_linear_model(), load_classifier()
    beta = dict(zip(model["columns"], model["beta"]))
    vendors = load_vendors().set_index("vendor_id")
    reqs = load_requirements()
    sheet = pd.read_csv(SHEET, encoding="utf-8-sig", dtype={"buyer_pincode": str},
                        keep_default_na=False)
    rationale = pd.read_csv(RATIONALES, encoding="utf-8").set_index("pair_id").rationale
    rng = np.random.default_rng(SEED)

    per_req, elig_info, scored_all = {}, {}, []
    for rid in sorted(test_ids):
        req = requirement_from_id(reqs, rid)
        s = sheet[sheet.requirement_id == rid]
        pool = pair_frame(req, vendors.loc[s.vendor_id].reset_index())
        pool["expected_score"] = s.set_index("vendor_id").expected_score.astype(int).loc[
            pool.vendor_id].to_numpy()
        per_req[rid], elig_info[rid], scored = evaluate_requirement(req, pool, model, clf, rng)
        scored_all.append(scored.assign(
            requirement_id=rid, needed_category=req["needed_category"],
            max_distance_km=req["max_distance_km"],
            preference_keyword=req["preference_keyword"],
            pair_id=scored.vendor_id.map(s.set_index("vendor_id").pair_id)))
    scored_all = pd.concat(scored_all, ignore_index=True)
    strata = pd.read_csv(PAIRS).set_index("pair_id").stratum
    scored_all["stratum"] = scored_all.pair_id.map(strata)
    eligible_by_stratum = (scored_all[scored_all.eligible].stratum.value_counts()
                           .sort_index().to_dict())
    relevant_by_stratum = (scored_all[scored_all.expected_score >= RELEVANT_AT].stratum
                           .value_counts().sort_index().to_dict())

    # random context: many permutations, to show how noisy one seed is
    ctx = []
    rng_ctx = np.random.default_rng(SEED)
    for rid in sorted(test_ids):
        s = sheet[sheet.requirement_id == rid]
        rel = (s.expected_score.astype(int) >= RELEVANT_AT).to_numpy().astype(int)
        ctx.append([ndcg_at_k(rel[rng_ctx.permutation(len(rel))]) for _ in range(N_RANDOM_CONTEXT)])
    random_ctx = float(np.mean(ctx))

    summary = {m: {"mean_ndcg@5": float(np.mean([per_req[r][m]["ndcg@5"] for r in per_req])),
                   "mean_precision@5": float(np.mean([per_req[r][m]["precision@5"]
                                                      for r in per_req]))}
               for m in METHODS}
    excl = sum(e["relevant_excluded_by_eligibility"] for e in elig_info.values())
    n_rel = sum(e["n_relevant"] for e in elig_info.values())
    for m in ["pipeline", "nearest"]:
        summary[m]["relevant_excluded_by_eligibility"] = excl
    summary["random"]["relevant_excluded_by_eligibility"] = None

    scored_all["error"] = scored_all.display_score - scored_all.expected_score
    worst = scored_all.reindex(scored_all.error.abs().sort_values(ascending=False,
                                                                   kind="mergesort").index).head(5)
    worst_rows = [{
        "pair_id": r.pair_id, "requirement_id": r.requirement_id,
        "needed_category": r.needed_category, "vendor_id": r.vendor_id, "title": r.title,
        "vendor_categories": r.categories, "distance_km": float(r.distance_km),
        "max_distance_km": float(r.max_distance_km), "eligible": bool(r.eligible),
        "failed_rules": r.failed_rules, "expected_score": int(r.expected_score),
        "raw_score": float(r.raw_score), "display_score": float(r.display_score),
        "error_display_minus_expected": float(r.error),
        "reason": error_reason(r, beta), "judge_rationale": rationale[r.pair_id]}
        for r in worst.itertuples()]

    metrics = {
        "phase": 2,
        "target_note": "Relevance uses the AI-judged expected_score (not ground truth).",
        "setup": {"test_requirements": sorted(test_ids), "pairs_per_requirement": 15,
                  "n_pairs": len(scored_all), "relevant_threshold": RELEVANT_AT, "k": K,
                  "n_relevant": n_rel, "random_seed": SEED,
                  "eligibility_rules": "R1 category + R2 distance (R3 keyword not required)",
                  "pool": "each method re-ranks the 15 annotated vendors of the requirement",
                  "eligible_pairs_by_sampling_stratum": eligible_by_stratum,
                  "relevant_pairs_by_sampling_stratum": relevant_by_stratum},
        "summary": summary,
        "random_mean_ndcg@5_over_1000_permutations": random_ctx,
        "per_requirement": {rid: {**{m: {k: v for k, v in per_req[rid][m].items()}
                                     for m in METHODS}, "eligibility": elig_info[rid]}
                            for rid in per_req},
        "worst_pipeline_errors": worst_rows,
        "score_error_test_pairs": {
            "MAE_display_vs_expected": float(scored_all.error.abs().mean()),
            "n_raw_below_0": int((scored_all.raw_score < 0).sum()),
            "n_raw_above_100": int((scored_all.raw_score > 100).sum())},
    }
    OUT_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_md(metrics, reqs), encoding="utf-8")
    for m in METHODS:
        print(f"{m:9s} NDCG@5 {summary[m]['mean_ndcg@5']:.4f}  P@5 "
              f"{summary[m]['mean_precision@5']:.4f}")
    print(f"relevant pairs {n_rel}; excluded by eligibility {excl}; random NDCG@5 mean over "
          f"{N_RANDOM_CONTEXT} permutations {random_ctx:.4f}")
    for w in worst_rows:
        print(f"  {w['pair_id']} {w['requirement_id']} {w['title']!r}: expected "
              f"{w['expected_score']}, shown {w['display_score']:.1f} -> {w['reason']}")
    print(f"wrote {OUT_JSON}, {OUT_MD}")


def render_md(m, reqs):
    s, st = m["summary"], m["setup"]
    req_info = reqs.set_index("requirement_id")
    lines = [
        "# Phase 2 results: vendor-matching prototype",
        "",
        "Generated by `src/phase2_evaluation.py`. Every number below comes from that run.",
        "",
        "> **Caveat.** Relevance uses the **AI-judged `expected_score`** (Claude Code against "
        "the rubric), not human ground truth. The match-score model was trained on the same kind "
        "of AI-judged scores.",
        "",
        "## 1. Setup",
        "",
        f"- Test requirements: {', '.join(st['test_requirements'])} (the 6 held out of the "
        "Phase 1 linear model; it never saw them).",
        f"- Each has 15 annotated vendors, {st['n_pairs']} pairs in all. Each method re-ranks "
        "exactly those 15 (only they have an expected_score).",
        f"- Relevant = expected_score ≥ {st['relevant_threshold']}: {st['n_relevant']} of "
        f"{st['n_pairs']} pairs.",
        "- Eligibility: R1 category and R2 distance; R3 is off because no keyword is required.",
        "- Methods: **pipeline** (eligible first, then predicted score), **nearest** (eligible "
        "first, then distance), **random** (one permutation per requirement, seed 42).",
        "",
        "## 2. Results (mean over the 6 requirements)",
        "",
        "| method | NDCG@5 | precision@5 | relevant vendors excluded by eligibility |",
        "|---|---|---|---|",
    ]
    for name in METHODS:
        ex = s[name]["relevant_excluded_by_eligibility"]
        lines.append(f"| {name} | {s[name]['mean_ndcg@5']:.4f} | "
                     f"{s[name]['mean_precision@5']:.4f} | "
                     f"{'n/a (no rules)' if ex is None else ex} |")
    lines += [
        "",
        f"For context, random ranking averaged over {N_RANDOM_CONTEXT} permutations gives "
        f"NDCG@5 {m['random_mean_ndcg@5_over_1000_permutations']:.4f}; one seed is noisy.",
        "",
        "Precision@5 is capped below 1 when a requirement has fewer than 5 relevant vendors.",
        "",
        "**Read with care: the judged pool makes eligibility easy.** The 15 vendors per "
        "requirement were sampled in Phase 1 by stratum (5 nearest in category, 4 in category "
        "but beyond the radius, 3 nearest other-category, 3 random). Eligible pairs by stratum: "
        + ", ".join(f"{k} {v}" for k, v in st["eligible_pairs_by_sampling_stratum"].items())
        + "; relevant pairs by stratum: "
        + ", ".join(f"{k} {v}" for k, v in st["relevant_pairs_by_sampling_stratum"].items())
        + ". So the rules mostly recover the sampling design, and the pipeline-vs-nearest "
        "difference comes from ordering within the eligible vendors and from which "
        "ineligible vendors come next. Precision@5 is identical for the two because both put "
        "the same eligible vendors in the top 5.",
        "",
        "### Per requirement",
        "",
        "| req | category | km | relevant | eligible | pipeline NDCG@5 | nearest NDCG@5 | "
        "random NDCG@5 | pipeline P@5 | nearest P@5 | random P@5 | relevant excluded |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rid, d in m["per_requirement"].items():
        e = d["eligibility"]
        r = req_info.loc[rid]
        lines.append(
            f"| {rid} | {r.needed_category} | {r.max_distance_km} | {e['n_relevant']} | "
            f"{e['n_eligible']} | {d['pipeline']['ndcg@5']:.4f} | {d['nearest']['ndcg@5']:.4f} | "
            f"{d['random']['ndcg@5']:.4f} | {d['pipeline']['precision@5']:.2f} | "
            f"{d['nearest']['precision@5']:.2f} | {d['random']['precision@5']:.2f} | "
            f"{e['relevant_excluded_by_eligibility']} |")
    excluded = [(rid, x) for rid, d in m["per_requirement"].items()
                for x in d["eligibility"]["relevant_excluded"]]
    lines += ["", "### Relevant vendors excluded by the eligibility rules", ""]
    if excluded:
        for rid, x in excluded:
            lines.append(f"- {rid} · {x['title']} (expected {x['expected_score']}): "
                         f"{x['failed_rules']}")
    else:
        lines.append("None.")
    lines += ["", "## 3. The 5 worst pipeline errors (|displayed score − expected_score|)", "",
              "| pair | req | vendor | listed categories | eligible | expected | raw | shown | "
              "reason | judge's rationale |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for w in m["worst_pipeline_errors"]:
        lines.append(
            f"| {w['pair_id']} | {w['requirement_id']} ({w['needed_category']}) | {w['title']} | "
            f"{w['vendor_categories'].replace(';', ', ')} | {'yes' if w['eligible'] else 'no'} | "
            f"{w['expected_score']} | {w['raw_score']:.2f} | {w['display_score']:.2f} | "
            f"{w['reason']} | {w['judge_rationale']} |")
    e = m["score_error_test_pairs"]
    lines += [
        "",
        f"Across all {st['n_pairs']} test pairs: mean |shown − expected| = "
        f"{e['MAE_display_vs_expected']:.4f}; {e['n_raw_below_0']} raw predictions are below 0 "
        f"and {e['n_raw_above_100']} above 100 (both clipped for display).",
        "",
        "## 4. Limitations",
        "",
        "- **AI-judged target.** Relevance and the training target are AI-judged scores, not "
        "human ground truth; human validation was not done.",
        "- **Noisy category labels.** R1 trusts source-file categories; mislabelled vendors "
        "pass R1 and get the category coefficient (see the worst errors).",
        "- **Snapshot dataset.** One 2026-09-27 snapshot; no opening hours, stock or prices.",
        "- **Clipped scores.** Displayed scores are clipped to 0-100; raw values are kept.",
        "- **Category check only for pharmacy.** Other categories have no classifier.",
        "- **Small evaluation.** 6 requirements × 15 judged vendors; the full candidate lists the "
        "app shows (up to hundreds of vendors) are mostly unjudged.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
