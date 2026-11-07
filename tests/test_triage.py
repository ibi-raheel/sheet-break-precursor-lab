import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import triage as tg

RAW = os.path.join(ROOT, "data", "raw", "processminer.csv")
RESULTS = os.path.join(ROOT, "src", "data", "results.json")


def test_rule_labels():
    # passes per episode -> keep; passes per break only -> one-episode artefact; neither -> noise
    assert tg.rule_label({"survives": True, "survives_ep": True}) == tg.HOLDS
    assert tg.rule_label({"survives": False, "survives_ep": True}) == tg.HOLDS
    assert tg.rule_label({"survives": True, "survives_ep": False}) == tg.ARTEFACT
    assert tg.rule_label({"survives": False, "survives_ep": False}) == tg.NOISE
    assert tg.ACTION[tg.ARTEFACT] == "drop"


@pytest.mark.skipif(not (os.path.exists(RAW) and os.path.exists(RESULTS)), reason="needs the raw CSV (DATA.md)")
def test_triage_reproduces_pipeline_results_on_real_data(tmp_path):
    R = json.load(open(RESULTS))
    o = tg.main([RAW, "--model", "--out", str(tmp_path / "t.json")])

    assert o["cost"]["events"] == R["meta"]["breaks"] == 124
    assert o["cost"]["line_hours_per_day"] == pytest.approx(R["downtime"]["hours_per_day"], abs=1e-9)
    assert o["episodes"]["episodes"] == R["episodes"]["all_episodes"] == 69
    assert o["episodes"]["follow_on"] == R["episodes"]["follow_on"] == 55

    k = o["clustering"]
    assert k["weibull_k"] == pytest.approx(R["reliability"]["weibull"]["k"], abs=1e-6)   # 0.87
    assert k["hawkes_branching"] == pytest.approx(R["reliability"]["hawkes"]["branching"], abs=1e-6)  # 0.40
    assert k["watch_window_min"] == R["reliability"]["followon_since_prev_min"]["q90"]   # 103 min
    assert k["verdict"].startswith("clustered")

    d = o["detectors"]
    assert d["rules"] == len(R["detectors"]) == 175
    assert d["survive_per_event"] == 1 and d["survive_per_episode"] == R["episode_level"]["survivors"] == 0
    x6 = next(x for x in d["list"] if x["id"] == "x6:level")
    assert x6["label"] == tg.ARTEFACT
    assert x6["q"] == pytest.approx(R["episode_level"]["top_break_level"]["q"], abs=1e-9)
    assert x6["q_ep"] == pytest.approx(R["episode_level"]["top_break_level"]["q_ep"], abs=1e-9)

    s = o["shap_vs_lift"]
    assert len(s["top10_failing"]) == 10 - R["top10_shap_survivors"] == 10
    assert s["spearman"] == pytest.approx(R["shap_lift_spearman"], abs=1e-9)

    m = o["first_break_model"]
    ap = R["hybrid"]["horizons"]["20"]["ap"]["sensor_gbm"]
    assert m["lead_ge_0"]["pr_auc"] == pytest.approx(ap["first"], abs=1e-9)        # 0.061
    assert m["lead_ge_0"]["base_rate"] == pytest.approx(ap["prev_first"], abs=1e-9)  # 0.0385
    assert m["verdict"] == "first-break signal above chance: screening only"
