"""Compare 2-4 vendors from the current Find-vendors result side by side."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULT_KEY, SCORE_LABEL, page_setup, rule_icon  # noqa: E402

from matching.features import FEATURES  # noqa: E402

page_setup("Compare")
st.title("Compare vendors")
res = st.session_state.get(RESULT_KEY)
if res is None:
    st.info("Run a search on **Find vendors** first; its results appear here.")
    st.stop()

allv = pd.concat([res.eligible, res.ineligible], ignore_index=True)
label = {r.vendor_id: (f"#{int(r.rank)} " if r.eligible else "✖ ") + f"{r.title} ({r.vendor_id})"
         for r in allv.itertuples()}
default = list(res.eligible.vendor_id[:2]) or list(allv.vendor_id[:2])
chosen = st.multiselect("Vendors (2-4)", list(label), default=default, format_func=label.get,
                        max_selections=4)
if len(chosen) < 2:
    st.info("Choose at least 2 vendors.")
    st.stop()

ex = [res.explanations[vid] for vid in chosen]
rows = allv.set_index("vendor_id").loc[chosen]
names = [f"{e['title']} ({e['vendor_id']})" for e in ex]

st.subheader("Score and features")
table = {"eligible": ["✅" if e["eligible"] else "❌" for e in ex],
         SCORE_LABEL: [round(e["display_score"], 2) for e in ex],
         "raw prediction": [round(e["raw_score"], 4) for e in ex]}
for f in FEATURES:
    table[f] = [e["features"][f] for e in ex]
st.dataframe(pd.DataFrame(table, index=names).T.astype(str), width="stretch")

st.subheader("Contributions (β × x)")
contrib = pd.DataFrame({n: {t["term"]: t["contribution"] for t in e["contributions"]}
                        for n, e in zip(names, ex)})
contrib.loc["sum = raw prediction"] = contrib.sum()
st.dataframe(contrib.round(4), width="stretch")

st.subheader("Rules, categories and address")
cols = st.columns(len(ex))
for col, e, (vid, r) in zip(cols, ex, rows.iterrows()):
    with col:
        st.markdown(f"**{e['title']}**")
        for rule in e["rules"]:
            st.markdown(f"{rule_icon(rule)} {rule['rule']}: {rule['detail']}")
        if e["category_check"] is not None:
            (st.warning if e["category_check"]["warning"] else st.info)(
                e["category_check"]["message"])
        st.markdown(f"**Categories:** {r.categories.replace(';', ', ')}")
        st.markdown(f"**Address:** {r.address or '(none)'}")
