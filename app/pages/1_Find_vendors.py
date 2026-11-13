"""Find vendors: pick an existing requirement or build a new one, then see the eligible
ranking, a map, per-vendor explanations and the not-eligible list."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (CHAIN_LABELS, RESULT_KEY, SCORE_COLUMN, get_matcher,  # noqa: E402
                    page_setup, render_explanation, result_table)

from matching.pipeline import make_requirement, requirement_from_id  # noqa: E402

page_setup("Find vendors")
st.title("Find vendors")
m = get_matcher()
v = m.vendors

# ---------------------------------------------------------------- requirement input
mode = st.radio("Requirement", ["Existing requirement", "New requirement"], horizontal=True)
if mode == "Existing requirement":
    reqs = m.requirements
    labels = {r.requirement_id: (f"{r.requirement_id} - {r.needed_category}, "
                                 f"{r.max_distance_km} km"
                                 + (f", keyword '{r.preference_keyword}'"
                                    if r.preference_keyword else ""))
              for r in reqs.itertuples()}
    c1, c2, c3 = st.columns([2, 1, 1])
    rid = c1.selectbox("Requirement", list(labels), format_func=labels.get)
    kw_req = c2.checkbox("Keyword required", value=False,
                         help="R3: the keyword must appear in the vendor title")
    chain = c3.selectbox("Chain preference", list(CHAIN_LABELS), format_func=CHAIN_LABELS.get)
    requirement = requirement_from_id(reqs, rid, kw_req, chain)
else:
    c1, c2 = st.columns(2)
    category = c1.selectbox("Category needed", m.categories)
    loc_mode = c2.radio("Location by", ["Pincode", "Locality", "Lat / lon"], horizontal=True)
    if loc_mode == "Pincode":
        pins = sorted(p for p in v.pincode.unique() if p)
        pin = c2.selectbox("Pincode", pins)
        area = v[v.pincode == pin]
        lat, lon = float(area.lat.mean()), float(area.lon.mean())
        c2.caption(f"Centroid of {len(area)} vendors in {pin}: {lat:.5f}, {lon:.5f}")
    elif loc_mode == "Locality":
        locs = sorted(v.locality.dropna().unique())
        loc = c2.selectbox("Locality", locs)
        area = v[v.locality == loc]
        lat, lon = float(area.lat.mean()), float(area.lon.mean())
        c2.caption(f"Centroid of {len(area)} vendors in {loc}: {lat:.5f}, {lon:.5f}")
    else:
        lat = c2.number_input("Latitude", value=float(round(v.lat.median(), 5)), format="%.5f")
        lon = c2.number_input("Longitude", value=float(round(v.lon.median(), 5)), format="%.5f")
    c3, c4, c5 = st.columns(3)
    dist = c3.slider("Max distance (km)", 1, 20, 3)
    kw = c4.text_input("Preference keyword (optional)", "")
    kw_req = c4.checkbox("Keyword required", value=False)
    chain = c5.selectbox("Chain preference", list(CHAIN_LABELS), format_func=CHAIN_LABELS.get)
    requirement = make_requirement(category, lat, lon, dist, kw, kw_req, chain)

res = m.match(requirement)
st.session_state[RESULT_KEY] = res
meta = res.meta

# ---------------------------------------------------------------- summary
c = st.columns(4)
c[0].metric("Candidates", meta["n_candidates"])
c[1].metric("Eligible", meta["n_eligible"])
c[2].metric("Not eligible", meta["n_ineligible"])
c[3].metric("Search radius (km)", f"{meta['search_radius_km']:g}")
if meta["extended"]:
    st.caption(f"Fewer than 10 eligible among the {meta['n_nearby']} nearest vendors, so "
               f"{meta['n_extension']} more vendors listed in {requirement['needed_category']} "
               "were added.")
if requirement["needed_category"] == "pharmacy":
    st.caption("Category check (pharmacy only): probability from the L2 pharmacy classifier. "
               "⚠️ marks < 0.5. Advisory only - it never changes eligibility or rank.")

# ---------------------------------------------------------------- results
st.subheader("Eligible vendors (ranked)")
if res.eligible.empty:
    st.warning("No vendor passes every rule. See the not-eligible list below.")
else:
    st.dataframe(result_table(res.eligible), hide_index=True, width="stretch",
                 column_config={"display_score": SCORE_COLUMN,
                                "distance_km": st.column_config.NumberColumn(
                                    "distance (km)", format="%.2f")})

# ---------------------------------------------------------------- map
st.subheader("Map")
pts = pd.concat([
    pd.DataFrame({"lat": [requirement["buyer_lat"]], "lon": [requirement["buyer_lon"]],
                  "color": ["#eb6834"], "size": [60.0]}),
    res.eligible.assign(color="#2a78d6", size=30.0)[["lat", "lon", "color", "size"]],
    res.ineligible.assign(color="#9a9a9a", size=15.0)[["lat", "lon", "color", "size"]]],
    ignore_index=True)
st.map(pts, latitude="lat", longitude="lon", color="color", size="size")
st.caption("Orange: buyer · Blue: eligible · Grey: not eligible")

# ---------------------------------------------------------------- explanations
st.subheader("Explanations")
for r in res.eligible.itertuples():
    with st.expander(f"#{r.rank} {r.title} - {r.display_score:.1f}"):
        render_explanation(res.explanations[r.vendor_id])

st.subheader("Not eligible")
if res.ineligible.empty:
    st.write("Every candidate is eligible.")
else:
    inel = result_table(res.ineligible, with_rank=False).assign(
        failed_rules=res.ineligible.failed_rules.to_numpy())
    st.dataframe(inel, hide_index=True, width="stretch",
                 column_config={"display_score": SCORE_COLUMN})
    pick = st.selectbox("Explain a not-eligible vendor", res.ineligible.vendor_id,
                        format_func=lambda vid: res.explanations[vid]["title"])
    with st.expander("Explanation", expanded=True):
        render_explanation(res.explanations[pick])
