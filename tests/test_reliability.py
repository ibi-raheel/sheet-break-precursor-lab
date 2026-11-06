import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import load as ld
from pipeline import reliability as rl
from pipeline import stats as st

RAW = os.path.join(ROOT, "data", "raw", "processminer.csv")
RESULTS = os.path.join(ROOT, "src", "data", "results.json")


# ---------------------------------------------------------------- episodes + BH
def test_episode_builder_on_hand_built_cluster():
    # breaks at 0, 30, 100 min chain (gaps 30, 70 < 120); 400 starts a new one;
    # 450 joins it; 1000 is alone.
    t = [0, 30, 100, 400, 450, 1000]
    ep, first = rl.episodes(t, 120)
    assert ep.tolist() == [0, 0, 0, 1, 1, 2]
    assert first.tolist() == [True, False, False, True, False, True]
    # a gap of exactly 120 starts a new episode ("less than 2 h" chains)
    assert rl.episodes([0, 120], 120)[0].tolist() == [0, 1]


def test_bh_on_clustered_pvalues_by_hand():
    # p = .01,.02,.03 (three breaks from one episode scored independently) and .5
    # q_i = p_(i) m / i: .04, .04, .04, .5
    q = st.bh_adjust([0.03, 0.5, 0.01, 0.02])
    assert q.tolist() == pytest.approx([0.04, 0.5, 0.04, 0.04])


# ---------------------------------------------------------------- downtime
def test_downtime_synthetic():
    # rows every 2 min; break at row 3; next row 20 min later -> 18 min downtime;
    # break at row 6 with no gap -> 0
    t = np.array([0, 2, 4, 6, 26, 28, 30, 32, 34], float)
    assert rl.post_break_downtime(t, [3, 6]).tolist() == [18.0, 0.0]


@pytest.mark.skipif(not os.path.exists(RAW), reason="raw data not downloaded")
def test_downtime_matches_hand_count_on_first_three_breaks():
    # Hand count from the raw CSV:
    #  break 1 at 05-01 08:38, next row 10:50 -> gap 132 min -> 130 min down
    #  break 2 at 05-02 04:02, next row 04:28 -> gap  26 min ->  24 min down
    #  break 3 at 05-02 06:24, next row 06:28 -> gap   4 min ->   2 min down
    df = ld.load(RAW)
    on = ld.break_onsets(df["y"])
    tmin = (df["time"] - df["time"].iloc[0]).dt.total_seconds().values / 60
    assert rl.post_break_downtime(tmin, on[:3]).tolist() == [130.0, 24.0, 2.0]


# ---------------------------------------------------------------- reliability models
def test_weibull_mle_recovers_shape_and_reduces_to_exponential():
    rng = np.random.default_rng(0)
    tau = 300 * rng.weibull(0.8, size=20000)
    m = rl.fit_weibull(tau)
    assert m["k"] == pytest.approx(0.8, abs=0.02)
    assert m["lam"] == pytest.approx(300, rel=0.03)
    e = rl.fit_exponential(tau)
    assert e["lam"] == pytest.approx(tau.mean())
    assert m["ll"] >= e["ll"]                  # nested model can only fit better


def test_weibull_k1_is_memoryless():
    # k = 1: P(break in next h | survived s) = 1 - exp(-h/lam), independent of s
    p = rl.weibull_p_next([0, 50, 500], 20, 1.0, 100.0)
    assert p == pytest.approx([1 - np.exp(-0.2)] * 3)
    # k < 1: risk falls with time since the last break
    p = rl.weibull_p_next([0, 50, 500], 20, 0.8, 100.0)
    assert p[0] > p[1] > p[2]


def test_hawkes_loglik_hand_calculation():
    # events 0 and 10, window to T = 20, mu=.1, a=.5, b=.1
    # log lambda(10) = log(.1 + .5 e^-1)
    # compensator on (0, 20] = .1*20 + (.5/.1) [(1-e^-2) + (1-e^-1)]
    mu, a, b = 0.1, 0.5, 0.1
    hand = np.log(mu + a * np.exp(-1)) - (mu * 20 + (a / b) * ((1 - np.exp(-2)) + (1 - np.exp(-1))))
    assert rl.hawkes_ll(mu, a, b, [0.0, 10.0], 20.0) == pytest.approx(hand)


def _simulate_hawkes(mu, a, b, T, rng):
    t, ev, lam_bar = 0.0, [], mu
    while True:
        lam_bar = mu + sum(a * np.exp(-b * (t - s)) for s in ev[-200:])
        t += rng.exponential(1 / lam_bar)
        if t > T:
            return np.array(ev)
        lam_t = mu + sum(a * np.exp(-b * (t - s)) for s in ev[-200:])
        if rng.uniform() * lam_bar <= lam_t:
            ev.append(t)


def test_hawkes_fit_recovers_branching_ratio():
    rng = np.random.default_rng(3)
    ev = _simulate_hawkes(mu=0.02, a=0.05, b=0.1, T=60000, rng=rng)   # branching 0.5
    m = rl.fit_hawkes(ev, 60000)
    assert m["branching"] == pytest.approx(0.5, abs=0.1)


def test_hawkes_p_next_by_hand():
    mu, a, b, h = 0.01, 0.02, 0.05, 20
    # no past events -> Poisson baseline
    assert rl.hawkes_p_next([5.0], [10.0], h, mu, a, b)[0] == pytest.approx(1 - np.exp(-mu * h))
    # one event at 10, row at 30: Lambda = mu h + (a/b) e^{-b 20} (1 - e^{-b h})
    lam = mu * h + (a / b) * np.exp(-b * 20) * (1 - np.exp(-b * h))
    assert rl.hawkes_p_next([30.0], [10.0], h, mu, a, b)[0] == pytest.approx(1 - np.exp(-lam))


# ---------------------------------------------------------------- bootstrap + alarms + economics
def test_mbb_indices_are_contiguous_blocks():
    ix = rl.mbb_indices(100, 10, np.random.default_rng(0))
    assert len(ix) == 100 and ix.max() < 100
    blocks = ix.reshape(10, 10)
    assert (np.diff(blocks, axis=1) == 1).all()


def test_alarm_events_and_warning_by_hand():
    above = np.zeros(40, bool)
    above[[12, 13]] = True      # starts at 12; break at 20 is 8 rows later -> true alarm
    above[30] = True            # no break in (30, 40] -> false alarm
    starts, true = rl.alarm_events(above, [20], W=10)
    assert starts.tolist() == [12, 30] and true.tolist() == [True, False]
    w, lead = rl.warned(above, [20], W=10, min_lead=5)
    assert w.tolist() == [True] and lead.tolist() == [8]     # 8 rows = 16 min ahead
    w2, _ = rl.warned(above, [15], W=10, min_lead=5)          # fires 3 rows ahead only
    assert w2.tolist() == [False]


def test_breakeven_by_hand():
    # recall .5, 4 breaks/day, 20 min down per break -> 40 min/day at e = 1
    # 3 false alarms/day x 5 min = 15 min/day -> break-even e = 15/40 = .375
    # e = .5: (20 - 15) min/day x 30 days / 60 = 2.5 line-hours/month
    r = rl.breakeven(0.5, 4, 20, 3, 5, effs=(0.5,))
    assert r["breakeven_e"] == pytest.approx(0.375)
    assert r["sweep"][0]["net_line_hours_per_month"] == pytest.approx(2.5)


def test_threshold_for_fa_hits_target():
    rng = np.random.default_rng(0)
    s = rng.normal(size=4800)
    thr = rl.threshold_for_fa(s, np.ones(4800, bool), np.array([], int), 10, shifts=20, target=1.0)
    starts, true = rl.alarm_events(s > thr, np.array([], int), 10)
    assert len(starts) <= 20 and len(starts) >= 15


# ---------------------------------------------------------------- exported numbers
@pytest.mark.skipif(not os.path.exists(RESULTS), reason="run the pipeline first")
def test_exported_reliability_and_downtime_are_consistent():
    r = json.load(open(RESULTS))
    e = r["episodes"]
    assert e["first_in_episode"] + e["follow_on"] == e["all_breaks"] == r["meta"]["breaks"]
    d = r["downtime"]
    assert d["gaps_within_2_rows_after_break"] <= d["gaps_total"] == r["meta"]["gaps"]
    assert d["total_hours"] == pytest.approx(d["mean_min_per_break"] * r["meta"]["breaks"] / 60)
    rel = r["reliability"]
    assert rel["hawkes"]["branching"] < 1                        # stationary
    assert rel["weibull"]["k_ci"][0] < rel["weibull"]["k"] < rel["weibull"]["k_ci"][1]
    for b in r["bootstrap"].values():
        assert b["lo"] <= b["point"] <= b["hi"]
