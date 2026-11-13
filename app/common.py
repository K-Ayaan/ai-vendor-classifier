"""Shared helpers for the Streamlit pages. Data comes only from data/processed/ and
artifacts/ (via matching.Matcher); headline numbers are read from reports/*.json."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matching.pipeline import Matcher  # noqa: E402
from matching.scoring import SCORE_LABEL  # noqa: E402

PHASE1_METRICS = ROOT / "reports/metrics.json"
PHASE2_METRICS = ROOT / "reports/phase2_metrics.json"
RESULT_KEY = "match_result"
CHAIN_LABELS = {"none": "No preference", "exclude": "Rank chains last",
                "prefer": "Rank chains first"}

CAVEATS = [
    "**estimated target.** The match-score model was trained on scores judged on"
    "human values.",
    "**Noisy categories.** A vendor's categories come from which source file it appeared in "
    "(e.g. a toll plaza listed under street markets). Eligibility rule R1 trusts these labels.",
    "**Snapshot.** One Google Maps snapshot (2026-09-27); shops may have moved or closed.",
    "**Clipped scores.** The linear model can predict below 0 or above 100; the displayed "
    "score is clipped to 0-100 and the raw value is shown in each explanation.",
    "**Category check only for pharmacy.** A classifier exists only for is_pharmacy; it is "
    "advisory and never changes eligibility.",
]


def page_setup(title):
    st.set_page_config(page_title=f"Vendor matching - {title}", layout="wide")


@st.cache_resource(show_spinner="Loading vendors and models...")
def get_matcher():
    return Matcher()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def rule_icon(r):
    return "⏭️" if r["skipped"] else ("✅" if r["passed"] else "❌")


def render_explanation(e):
    """Rules, contributions and category check for one vendor (all real values)."""
    st.markdown("**Eligibility rules**")
    for r in e["rules"]:
        st.markdown(f"{rule_icon(r)} **{r['rule']}**: {r['detail']}")
    cc = e["category_check"]
    if cc is not None:
        (st.warning if cc["warning"] else st.info)(cc["message"])
    st.markdown(f"**Score breakdown**: raw prediction {e['raw_score']:.2f} "
                f"(displayed {e['display_score']:.1f} after clipping to 0-100)")
    st.dataframe(contributions_table(e), hide_index=True, width="stretch")


def contributions_table(e):
    t = pd.DataFrame(e["contributions"])
    t.loc[len(t)] = ["sum = raw prediction", None, None, t.contribution.sum()]
    return t.rename(columns={"term": "term", "coefficient": "coefficient β",
                             "value": "feature value x", "contribution": "β × x"})


def result_table(df, with_rank=True):
    cols = (["rank"] if with_rank else []) + ["title", "distance_km", "display_score"]
    out = df[cols].copy()
    out["eligibility"] = ["✅ Eligible" if e else "❌ Not eligible" for e in df.eligible]
    if "pharmacy_probability" in df:
        out["category check"] = [f"⚠️ {p:.2f}" if p < 0.5 else f"{p:.2f}"
                                 for p in df.pharmacy_probability]
    return out


SCORE_COLUMN = st.column_config.ProgressColumn(SCORE_LABEL, min_value=0, max_value=100,
                                               format="%.1f")
