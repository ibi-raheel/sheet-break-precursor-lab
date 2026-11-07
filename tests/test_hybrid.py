import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import hybrid as hy

RESULTS = os.path.join(ROOT, "src", "data", "results.json")


def test_budget_alarm_and_warning_by_hand():
    # 10 eligible rows, budget 20% -> the 2 highest scores (rows 3 and 7) alarm.
    s = np.array([.1, .2, .3, .9, .1, .2, .1, .8, .0, .0])
    el = np.ones(10, bool)
    a = hy.budget_alarm(s, el, 0.20)
    assert np.where(a)[0].tolist() == [3, 7]
    # ineligible rows never alarm and do not count toward the budget
    el2 = el.copy()
    el2[3] = False
    assert np.where(hy.budget_alarm(s, el2, 0.20))[0].tolist() == [2, 7]  # 9 rows -> k=2: .8, .3
    # ties at the cut: exactly k rows alarm, earlier rows first
    assert np.where(hy.budget_alarm(np.array([.5, .5, .5, .1]), np.ones(4, bool), 0.5))[0].tolist() == [0, 1]
    # break onsets at rows 5 and 9; window W=3 rows, lead >= 2 rows:
    # onset 5 is warned if the alarm fires in rows 2..3 -> row 3 fires -> warned.
    # onset 9 needs a firing in rows 6..7 -> row 7 fires -> warned.
    # with lead >= 3: onset 5 needs a firing at row 2, onset 9 at row 6 -> neither.
    assert hy.warned_by(a, [5, 9], 3, 2).tolist() == [True, True]
    assert hy.warned_by(a, [5, 9], 3, 3).tolist() == [False, False]
    # rising edges at 3 and 7; next onsets 5 (2 rows ahead) and 9 (2 ahead): no false alarms
    assert hy.false_alarms(a, [5, 9], 3) == (0, 2)
    assert hy.false_alarms(a, [9], 3) == (1, 2)   # row 3 has no onset within 3 rows


def test_offset_gbm_with_zero_learning_returns_the_hazard():
    # With a vanishing learning rate the trees add nothing, so the hybrid equals the hazard.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (rng.random(400) < 0.1).astype(int)
    hz = rng.uniform(0.01, 0.3, 400)
    m = hy.fit_offset_gbm(X, y, hy.logit(hz), seed=0, n_jobs=1, learning_rate=1e-12, n_estimators=5,
                          min_child_samples=5)
    assert hy.predict_offset_gbm(m, X, hy.logit(hz)) == pytest.approx(hz, rel=1e-6)


def test_offset_gbm_learns_a_sensor_term_the_hazard_lacks():
    # True risk = hazard x 6 when sensor x0 > 1. The hybrid should rank those rows higher.
    rng = np.random.default_rng(1)
    n = 6000
    X = rng.normal(size=(n, 2))
    hz = np.full(n, 0.02)
    p = np.where(X[:, 0] > 1, 0.12, 0.02)
    y = (rng.random(n) < p).astype(int)
    m = hy.fit_offset_gbm(X, y, hy.logit(hz), seed=0, n_jobs=1)
    ph = hy.predict_offset_gbm(m, X, hy.logit(hz))
    assert ph[X[:, 0] > 1.5].mean() > 3 * ph[X[:, 0] < 0].mean()


def test_episode_safe_bounds_never_cut_an_episode():
    t = np.arange(0, 3000, 2.0)            # 1500 rows, 2-minute sampling
    onsets = np.array([740, 745, 760])     # one episode around row 750 (minute 1500)
    b = hy.episode_safe_bounds(len(t), onsets, t, 2, gap_min=120)
    # the midpoint (row 750) is inside the episode; the boundary must sit >= 120 min
    # (60 rows) from every onset: nearest allowed row is 680 (|1360-1480| = 120)
    assert b.tolist() == [0, 680, 1500]


def test_cluster_bootstrap_by_hand():
    # identical outcomes -> difference exactly 0 with a zero-width interval
    a = np.array([1, 0, 1, 1])
    g = np.array([0, 0, 1, 2])
    r = hy.cluster_boot_diff(a, a, g, 200, np.random.default_rng(0))
    assert r["point"] == 0 and r["lo"] == 0 and r["hi"] == 0
    # a beats b on every item -> every resample is +1
    r = hy.cluster_boot_diff(np.ones(4), np.zeros(4), g, 200, np.random.default_rng(0))
    assert r["lo"] == 1 and r["share_above_0"] == 1.0


def test_subset_ap_restricts_positives_only():
    y = np.array([0, 1, 0, 1, 0])
    s = np.array([.1, .9, .2, .3, .4])
    nf = np.array([False, True, False, False, False])  # row 1's next break is a first break
    r = hy.subset_ap(y, s, nf)
    # first: rows 0,1,2,4 -> the positive ranks top -> AP 1
    assert r["first"] == pytest.approx(1.0)
    # follow: rows 0,2,3,4; positive (.3) ranked 2nd behind .4 -> AP = 1/2
    assert r["follow"] == pytest.approx(0.5)


@pytest.mark.skipif(not os.path.exists(RESULTS), reason="results not built")
def test_exported_hybrid_results_are_consistent():
    r = json.load(open(RESULTS))
    hyb = r["hybrid"]
    for hm, ev in hyb["horizons"].items():
        # matched budgets: every model's alarm is on for about the same share of time
        for bgt, row in ev["budgets"].items():
            shares = [m["on_share"] for m in row.values()]
            assert max(shares) - min(shares) < 0.01, (hm, bgt, shares)
            assert all(abs(s - float(bgt)) < 0.01 for s in shares)
        assert ev["test_first_breaks"] + ev["test_follow_breaks"] == ev["test_breaks"]
        for v in ev["ap_ci"].values():
            for c in v.values():
                assert c["lo"] <= c["hi"] and c["B"] >= 1000


def test_shift_null_shows_flicker_hits_windows_by_chance():
    # 200 rows, one block, alarm budget 10% (20 rows); breaks at rows 50 and 150,
    # warning window 4 rows with lead >= 1 (rows o-4..o-1).
    # A comb alarm (every 10th row) hits any 4-row window with probability 4/10;
    # one 20-row block hits it with probability (20 + 4 - 1) / 200 = 0.115.
    n = 200
    idx = [np.arange(n)]
    onsets = np.array([50, 150])
    groups = np.array([True, True])
    comb = (np.arange(n) % 10 == 0).astype(float)
    block = (np.arange(n) < 20).astype(float)
    rng = np.random.default_rng(0)
    nc = hy.shift_null_warned(comb, idx, 0.10, onsets, groups, 4, 1, 400, rng)[True]
    nb = hy.shift_null_warned(block, idx, 0.10, onsets, groups, 4, 1, 400, rng)[True]
    assert nc.mean() == pytest.approx(2 * 0.4, abs=0.1)
    assert nb.mean() == pytest.approx(2 * 0.115, abs=0.07)


def test_superpose_is_sum_of_poisson_intensities_by_hand():
    # Excitation gives P = 0.2, sensor background gives P = 0.5 over the horizon.
    # Superposed: 1 - 0.8 * 0.5 = 0.6.
    assert hy.superpose(0.2, 0.5) == pytest.approx(0.6)
    # Same thing via intensities: Lambda_a = -ln 0.8, Lambda_b = -ln 0.5,
    # P = 1 - exp(-(Lambda_a + Lambda_b)) = 1 - 0.8 * 0.5.
    la, lb_ = -np.log(0.8), -np.log(0.5)
    assert hy.superpose(0.2, 0.5) == pytest.approx(1 - np.exp(-(la + lb_)))
    # A zero background returns the excitation term unchanged, and vice versa.
    p = np.array([0.0, 0.1, 0.7])
    assert np.allclose(hy.superpose(p, 0.0), p)
    assert np.allclose(hy.superpose(0.0, p), p)


def test_superposed_hazard_with_zero_background_is_excitation_only_hawkes():
    # Hawkes with mu = 0 plus a background of P_bg = 1 - exp(-mu h) is the full Hawkes.
    from pipeline import reliability as rl
    rt = np.linspace(0, 600, 301)
    ev = np.array([50.0, 60.0, 300.0])
    mu, a, b, h = 0.003, 0.004, 0.02, 20
    full = rl.hawkes_p_next(rt, ev, h, mu, a, b)
    exc = rl.hawkes_p_next(rt, ev, h, 0.0, a, b)
    assert np.allclose(hy.superpose(exc, 1 - np.exp(-mu * h)), full)


def test_shift_null_ap_separates_aligned_from_rotated_scores():
    # Labels: 1 on rows 45..49 and 145..149 of 200. A score equal to the label has AP 1;
    # rotating it inside the block misaligns it, so the null mean is far below 1.
    n = 200
    y = np.zeros(n, int)
    y[45:50] = 1
    y[145:150] = 1
    rng = np.random.default_rng(0)
    nl = hy.shift_null_ap(y, y.astype(float) + 1e-3 * rng.random(n), [np.arange(n)], np.ones(n, bool), 300, rng)
    assert nl.mean() < 0.3
    assert (nl >= 0.999).mean() < 0.05
