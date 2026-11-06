import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import features as fe
from pipeline import label as lb
from pipeline import load as ld
from pipeline import stats as st


def test_lift_and_precision_match_hand_calculation():
    # Hand calculation (METHOD.md section 4): 8 of 74 pre-break windows fire,
    # 130 of 11740 nominal windows fire.
    # lift = (8/74) / (130/11740) = 0.108108 / 0.011073 = 9.7630
    # precision (window level) = 8 / (8 + 130) = 0.057971
    assert st.lift_from_counts(8, 74, 130, 11740) == pytest.approx(9.7630, abs=1e-3)
    assert st.precision_from_counts(8, 130) == pytest.approx(0.057971, abs=1e-5)


RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "data", "results.json")


@pytest.mark.skipif(not os.path.exists(RESULTS), reason="run the pipeline first")
def test_exported_top_detector_matches_hand_calculation():
    import json
    r = json.load(open(RESULTS))
    d = r["detectors"][0]
    n_ev, n_nom = r["meta"]["eval_breaks"], r["meta"]["nominal_windows"]
    fire_pre = round(d["coverage"] * n_ev)
    fire_nom = round(d["nominal_rate"] * n_nom)
    assert d["lift"] == pytest.approx(st.lift_from_counts(fire_pre, n_ev, fire_nom, n_nom), rel=1e-9)


def test_lift_simple_2x2():
    # 3/10 pre-break windows fire vs 5/100 nominal: lift = 0.3/0.05 = 6
    assert st.lift_from_counts(3, 10, 5, 100) == pytest.approx(6.0)


def test_perm_pvalue_formula():
    null = np.array([0.0] * 95 + [0.5] * 5)
    # 5 of 100 null draws >= 0.5 -> p = (1+5)/(1+100)
    assert st.perm_pvalue(0.5, null) == pytest.approx(6 / 101)
    assert st.perm_pvalue(0.9, null) == pytest.approx(1 / 101)


def test_bh_matches_scipy():
    from scipy.stats import false_discovery_control
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert np.allclose(st.bh_adjust(p), false_discovery_control(p, method="bh"))


def test_window_any_is_past_only():
    f = np.zeros(10, bool)
    f[5] = True
    wa = st.window_any(f, 3)
    assert wa.tolist() == [False] * 5 + [True, True, True] + [False, False]


def test_break_onsets_merge_consecutive_rows():
    y = [0, 1, 1, 0, 0, 1, 0, 1]
    assert ld.break_onsets(y).tolist() == [1, 5, 7]


def test_labels_forward_window_and_mask():
    label, elig = lb.make_labels(20, np.array([10]), h_rows=3, mask_k=2)
    assert label.tolist()[6:11] == [0, 1, 1, 1, 0]   # rows 7,8,9 precede onset 10
    assert not elig[10] and not elig[11] and not elig[12] and elig[13]
    shifted, _ = lb.make_labels(20, np.array([10]), h_rows=3, mask_k=2, extra_shift=1)
    assert shifted.tolist()[5:11] == [0, 1, 1, 1, 0, 0]


def test_features_use_only_past_rows():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.normal(size=60), "x2": rng.normal(size=60)})
    a = fe.build(df, ["x1", "x2"])
    df2 = df.copy()
    df2.loc[40:, ["x1", "x2"]] += 100.0      # change the future
    b = fe.build(df2, ["x1", "x2"])
    pd.testing.assert_frame_equal(a.iloc[:40], b.iloc[:40])


def test_slope_feature_recovers_known_slope():
    df = pd.DataFrame({"x1": 2.5 * np.arange(30, dtype=float)})
    F = fe.build(df, ["x1"])
    assert F["x1__slope5"].dropna().to_numpy() == pytest.approx(2.5)


def test_restart_mask_detects_transient_length():
    rng = np.random.default_rng(1)
    X = rng.normal(scale=0.01, size=(200, 4))
    # big oscillation for 5 rows after the break at row 100
    X[101:106] += np.array([[3], [-3], [3], [-3], [3]])
    k, lens = lb.restart_mask_len(X, np.array([100]), q=1.0)
    assert 5 <= k <= 7
