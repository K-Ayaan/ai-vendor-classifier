"""Overview page. Run with: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CAVEATS, PHASE1_METRICS, get_matcher, page_setup, read_json  # noqa: E402

from matching.features import category_set  # noqa: E402

page_setup("Overview")
st.title("Intelligent Agent Mapping: Vendor Matching")
st.caption("Phase 2 prototype: deterministic eligibility rules + the Phase 1 linear match-score "
           "model. Pages: **Find vendors**, **Compare**, **Evaluation** (sidebar).")

m = get_matcher()
v = m.vendors
p1 = read_json(PHASE1_METRICS)

c = st.columns(4)
c[0].metric("Vendors", f"{len(v):,}")
c[1].metric("Categories", len(m.categories))
c[2].metric("Requirements", len(m.requirements))
c[3].metric("Annotated pairs", p1["data"]["pairs"] if p1 else "n/a")

st.subheader("Caveats")
for text in CAVEATS:
    st.markdown(f"- {text}")

left, right = st.columns([1, 1])
with left:
    st.subheader("The 23 categories")
    counts = pd.Series([c for s in v.categories.map(category_set) for c in s]).value_counts()
    st.dataframe(counts.rename_axis("category").reset_index(name="vendors listed")
                 .sort_values("category"), hide_index=True, width="stretch", height=420)
with right:
    st.subheader("Phase 1 headline metrics")
    if p1 is None:
        st.info("reports/metrics.json not found - run the Phase 1 pipeline.")
    else:
        reg, clf = p1["regression"], p1["classification"]
        l2 = clf["l2_model"]["test_metrics"]
        rows = [
            ("Match score model: linear regression (Gaussian MLE)", "", ""),
            ("  target", reg["target"], ""),
            ("  R²", f"{reg['train']['R2']:.4f}", f"{reg['test']['R2']:.4f}"),
            ("  RMSE (score points)", f"{reg['train']['RMSE']:.2f}", f"{reg['test']['RMSE']:.2f}"),
            ("  train / test requirements", str(reg["split"]["train_requirements"]),
             str(reg["split"]["test_requirements"])),
            ("Pharmacy classifier: L2 logistic (MAP)", "", ""),
            ("  F1", "", f"{l2['f1']:.4f}"),
            ("  precision / recall", "", f"{l2['precision']:.4f} / {l2['recall']:.4f}"),
            ("  ROC-AUC", "", f"{l2['roc_auc']:.4f}"),
            ("  majority baseline accuracy", "",
             f"{clf['majority_baseline']['accuracy']:.4f}"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["quantity", "train", "test"]),
                     hide_index=True, width="stretch")
        coef = pd.Series(reg["coefficients"]).rename("β").rename_axis("term").reset_index()
        st.markdown("**Linear model coefficients** (the app uses exactly these)")
        st.dataframe(coef, hide_index=True, width="stretch")
