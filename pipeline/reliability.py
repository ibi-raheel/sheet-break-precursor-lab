"""Break clustering, reliability models, downtime and alarm economics.

Pure functions; run.py wires them to the data. Times are clock minutes from
the first timestamp unless stated otherwise.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score


# ---------------------------------------------------------------- episodes
def episodes(times_min, gap_min=120.0):
    """Chain breaks less than gap_min apart into episodes.

    Returns (episode_id per break, first_in_episode flag per break)."""
    t = np.asarray(times_min, float)
    if len(t) == 0:
        return np.array([], int), np.array([], bool)
    new = np.r_[True, np.diff(t) >= gap_min]
    return np.cumsum(new) - 1, new


# ---------------------------------------------------------------- downtime
def post_break_downtime(times_min, onsets, look=2, nominal=2.0):
    """Downtime after each break: the first timestamp gap within `look` rows
    after the onset, minus the nominal sampling step. 0 when no gap."""
    t = np.asarray(times_min, float)
    out = np.zeros(len(onsets))
    for i, o in enumerate(onsets):
        for r in range(o + 1, min(len(t), o + 1 + look)):
            d = t[r] - t[r - 1]
            if d > nominal:
                out[i] = d - nominal
                break
    return out


# ---------------------------------------------------------------- renewal fits
def _weib_ll(k, lam, tau, cens=None):
    z = tau / lam
    ll = np.sum(np.log(k) - np.log(lam) + (k - 1) * np.log(z) - z ** k)
    if cens is not None:
        ll -= (cens / lam) ** k
    return ll


def fit_exponential(tau, cens=None):
    tot = tau.sum() + (cens or 0.0)
    lam = tot / len(tau)                      # mean time between breaks
    return {"lam": float(lam), "ll": float(_weib_ll(1.0, lam, tau, cens)), "k_params": 1}


def fit_weibull(tau, cens=None):
    """MLE of Weibull(k, lam) on inter-break times, last interval right-censored.
    hazard h(t) = (k/lam) (t/lam)^(k-1)."""
    tau = np.asarray(tau, float)
    f = lambda p: -_weib_ll(np.exp(p[0]), np.exp(p[1]), tau, cens)
    r = minimize(f, x0=[0.0, np.log(tau.mean())], method="Nelder-Mead",
                 options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
    k, lam = np.exp(r.x)
    return {"k": float(k), "lam": float(lam), "ll": float(-r.fun), "k_params": 2}


def weibull_cdf(x, k, lam):
    return 1 - np.exp(-(np.asarray(x, float) / lam) ** k)


def weibull_p_next(since, h, k, lam):
    """P(break in (s, s+h] | no break in (0, s]) = 1 - S(s+h)/S(s)."""
    s = np.asarray(since, float)
    return 1 - np.exp(-(((s + h) / lam) ** k - (s / lam) ** k))


# ---------------------------------------------------------------- Hawkes
def hawkes_ll(mu, a, b, t, T):
    """Log-likelihood of a self-exciting process on (t_1, T], conditioned on the
    first event. lambda(t) = mu + sum_{t_i<t} a exp(-b (t - t_i))."""
    t = np.asarray(t, float)
    A = 0.0
    ll = 0.0
    for i in range(1, len(t)):
        A = np.exp(-b * (t[i] - t[i - 1])) * (1 + A)
        ll += np.log(mu + a * A)
    comp = mu * (T - t[0]) + (a / b) * np.sum(1 - np.exp(-b * (T - t)))
    return ll - comp


def fit_hawkes(t, T):
    t = np.asarray(t, float)
    rate = (len(t) - 1) / (T - t[0])
    best = None
    for b0 in (1 / 30, 1 / 120, 1 / 600):
        x0 = [np.log(rate * 0.5), np.log(0.5 * b0), np.log(b0)]
        r = minimize(lambda p: -hawkes_ll(*np.exp(p), t, T), x0=x0, method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 6000})
        if best is None or r.fun < best.fun:
            best = r
    mu, a, b = np.exp(best.x)
    return {"mu": float(mu), "a": float(a), "b": float(b), "branching": float(a / b),
            "ll": float(-best.fun), "k_params": 3}


def hawkes_p_next(row_times, event_times, h, mu, a, b):
    """P(event in (t, t+h] | history) for each row time, using only events <= t.
    Lambda = mu h + (a/b) E(t) (1 - exp(-b h)), E(t) = sum exp(-b (t - t_j))."""
    rt = np.asarray(row_times, float)
    ev = np.sort(np.asarray(event_times, float))
    E = np.zeros(len(rt))
    j, cur, tlast = 0, 0.0, None
    for i, t in enumerate(rt):
        while j < len(ev) and ev[j] <= t:
            cur = (cur * np.exp(-b * (ev[j] - tlast)) if tlast is not None else 0.0) + 1.0
            tlast = ev[j]
            j += 1
        E[i] = cur * np.exp(-b * (t - tlast)) if tlast is not None else 0.0
    lam_int = mu * h + (a / b) * E * (1 - np.exp(-b * h))
    return 1 - np.exp(-lam_int)


def aic(m):
    return 2 * m["k_params"] - 2 * m["ll"]


# ---------------------------------------------------------------- bootstrap
def mbb_indices(n, block, rng):
    """Moving-block bootstrap: concatenate random contiguous blocks to length n."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=nb)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def ap_diff_ci(y, s1, s2, block, B, rng, alpha=0.05):
    """CI for AP(s1) - AP(s2) on time-ordered rows by moving-block bootstrap."""
    y = np.asarray(y)
    diffs = []
    for _ in range(B):
        ix = mbb_indices(len(y), block, rng)
        if y[ix].sum() == 0:
            continue
        diffs.append(average_precision_score(y[ix], s1[ix]) - average_precision_score(y[ix], s2[ix]))
    d = np.array(diffs)
    return {"point": float(average_precision_score(y, s1) - average_precision_score(y, s2)),
            "lo": float(np.quantile(d, alpha / 2)), "hi": float(np.quantile(d, 1 - alpha / 2)),
            "share_above_0": float((d > 0).mean()), "B": int(len(d))}


# ---------------------------------------------------------------- event-level alarms
def alarm_events(above, onsets, W):
    """Rising edges of `above`; an edge is true if a break onset lies in (t, t+W]."""
    above = np.asarray(above, bool)
    starts = np.where(above & ~np.r_[False, above[:-1]])[0]
    on = np.sort(np.asarray(onsets))
    if len(on) == 0:
        return starts, np.zeros(len(starts), bool)
    pos = np.searchsorted(on, starts, side="right")
    nxt = np.where(pos < len(on), on[np.minimum(pos, len(on) - 1)], 10**12)
    true = (nxt - starts) <= W
    return starts, true


def warned(above, onsets, W, min_lead):
    """For each onset: did the alarm fire at a row t with min_lead <= o - t <= W?
    Returns (warned flags, lead rows of the earliest firing in [o-W, o) or -1)."""
    above = np.asarray(above, bool)
    w, lead = [], []
    for o in onsets:
        seg = above[max(0, o - W):o]
        idx = np.where(seg)[0]
        first = (len(seg) - idx[0]) if len(idx) else -1
        lead.append(first)
        w.append(bool(above[max(0, o - W):o - min_lead + 1].any()))
    return np.array(w), np.array(lead)


def threshold_for_fa(score, eligible, onsets, W, shifts, target):
    """Lowest threshold whose false-alarm rate (per shift) stays <= target,
    scanning down a quantile grid of the eligible scores."""
    s = np.where(eligible, score, -np.inf)
    vals = s[np.isfinite(s)]
    grid = np.unique(np.quantile(vals, np.linspace(0.5, 0.99999, 400)))[::-1]
    best = grid[0]
    for thr in grid:
        st_, tr_ = alarm_events(s > thr, onsets, W)
        if (len(st_) - tr_.sum()) / shifts > target:
            break
        best = thr
    return float(best)


def breakeven(recall, breaks_per_day, downtime_min, fa_per_day, check_min=5.0,
              effs=(0.10, 0.25, 0.50), days_per_month=30.0):
    """Net line-minutes saved per day = recall*breaks/day*downtime*e - FA/day*check.
    Break-even e = FA/day*check / (recall*breaks/day*downtime)."""
    gain_full = recall * breaks_per_day * downtime_min
    cost = fa_per_day * check_min
    be = cost / gain_full if gain_full > 0 else float("inf")
    sweep = [{"e": e, "net_line_hours_per_month": float((gain_full * e - cost) * days_per_month / 60)}
             for e in effs]
    return {"breakeven_e": float(be), "gross_min_per_day_at_e1": float(gain_full),
            "check_cost_min_per_day": float(cost), "sweep": sweep}
