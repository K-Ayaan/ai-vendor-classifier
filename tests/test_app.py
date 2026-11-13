"""Every page renders without exceptions under streamlit.testing AppTest."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from matching.pipeline import Matcher

TIMEOUT = 120
ROOT = Path(__file__).resolve().parents[1]


def run(path, state=None):
    at = AppTest.from_file(str(ROOT / path), default_timeout=TIMEOUT)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_overview_page():
    at = run("app/streamlit_app.py")
    assert at.title[0].value == "Intelligent Agent Mapping: Vendor Matching"
    assert [x.value for x in at.metric][:3] == ["4,612", "23", "30"]


def test_find_vendors_existing_requirement():
    at = run("app/pages/1_Find_vendors.py")
    assert len(at.dataframe) >= 2 and len(at.expander) > 0
    res = at.session_state["match_result"]
    assert res.requirement["requirement_id"] == "R01"


def test_find_vendors_pharmacy_shows_category_check():
    at = run("app/pages/1_Find_vendors.py")
    pharm = [o for o in at.selectbox[0].options if "pharmacy" in o]
    assert pharm
    at.selectbox[0].set_value(pharm[0].split(" - ")[0]).run()
    assert not at.exception
    assert any("Category check" in c.value for c in at.caption)


@pytest.mark.parametrize("loc", ["Pincode", "Locality", "Lat / lon"])
def test_find_vendors_new_requirement(loc):
    at = run("app/pages/1_Find_vendors.py")
    at.radio[0].set_value("New requirement").run()
    at.radio[1].set_value(loc).run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state["match_result"].requirement["requirement_id"] == "NEW"


def test_compare_without_result_shows_hint():
    at = run("app/pages/2_Compare.py")
    assert any("Find vendors" in i.value for i in at.info)


def test_compare_with_result():
    res = Matcher().match_id("R09")
    at = run("app/pages/2_Compare.py", {"match_result": res})
    assert len(at.multiselect[0].value) == 2 and len(at.dataframe) == 2


def test_evaluation_page():
    at = run("app/pages/3_Evaluation.py")
    assert len(at.dataframe) == 3
