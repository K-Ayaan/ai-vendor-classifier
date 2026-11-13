"""Evaluation: the Step 5 results from reports/phase2_metrics.json."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PHASE2_METRICS, page_setup, read_json  # noqa: E402

page_setup("Evaluation")
st.title("Evaluation on the 6 held-out requirements")
m = read_json(PHASE2_METRICS)
if m is None:
    st.info("reports/phase2_metrics.json not found - run `python src/phase2_evaluation.py`.")
    st.stop()

s = m["setup"]
st.warning("Relevance uses the **AI-judged expected_score** (not human ground truth).")
st.markdown(
    f"Requirements {', '.join(s['test_requirements'])} were held out of the linear model. "
    f"Each method re-ranks the 15 annotated vendors of each requirement ({s['n_pairs']} pairs). "
    f"Relevant = expected_score ≥ {s['relevant_threshold']} ({s['n_relevant']} pairs). "
    f"Eligibility: {s['eligibility_rules']}.")

rows = []
for name, d in m["summary"].items():
    ex = d["relevant_excluded_by_eligibility"]
    rows.append({"method": name, "NDCG@5": d["mean_ndcg@5"],
                 "precision@5": d["mean_precision@5"],
                 "relevant excluded by eligibility": "n/a (no rules)" if ex is None else ex})
st.subheader("Mean over the 6 requirements")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
             column_config={"NDCG@5": st.column_config.NumberColumn(format="%.4f"),
                            "precision@5": st.column_config.NumberColumn(format="%.4f")})
st.caption(f"Random over 1000 permutations: NDCG@5 "
           f"{m['random_mean_ndcg@5_over_1000_permutations']:.4f} (one seed is noisy).")

st.subheader("Per requirement")
per = []
for rid, d in m["per_requirement"].items():
    e = d["eligibility"]
    per.append({"requirement": rid, "relevant": e["n_relevant"], "eligible": e["n_eligible"],
                **{f"{k} NDCG@5": d[k]["ndcg@5"] for k in ["pipeline", "nearest", "random"]},
                **{f"{k} P@5": d[k]["precision@5"] for k in ["pipeline", "nearest", "random"]},
                "relevant excluded": e["relevant_excluded_by_eligibility"]})
st.dataframe(pd.DataFrame(per).round(4), hide_index=True, width="stretch")

st.subheader("5 worst pipeline errors")
w = pd.DataFrame(m["worst_pipeline_errors"])[
    ["pair_id", "requirement_id", "title", "eligible", "expected_score", "raw_score",
     "display_score", "reason", "judge_rationale"]]
st.dataframe(w.round(2), hide_index=True, width="stretch")
e = m["score_error_test_pairs"]
st.caption(f"Mean |shown − expected| over test pairs: {e['MAE_display_vs_expected']:.2f}; "
           f"raw below 0: {e['n_raw_below_0']}, above 100: {e['n_raw_above_100']}.")
