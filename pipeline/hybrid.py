"""Hybrid break model: self-exciting hazard baseline + learned sensor terms.

The hazard (Hawkes) model knows only break history. The learned part is a
gradient-boosted model that starts from the hazard's log-odds (LightGBM
`init_score`) and can only add or subtract risk from it, using sensor
features. With zero trees it returns the hazard exactly.

Pure functions; hybrid_run.py wires them to the data.
"""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


# ---------------------------------------------------------------- folds
def episode_safe_bounds(n, onset_rows, t_min, k, gap_min=120.0):
    """Split rows 0..n into k contiguous blocks. Each inner boundary is moved to
    the nearest row that is at least gap_min from every break onset, so no
    episode (breaks < gap_min apart) is cut in two."""
    t = np.asarray(t_min, float)
    on_t = t[np.asarray(onset_rows)]
    target = np.linspace(0, n, k + 1).astype(int)
    ok = np.ones(n, bool)
    for ot in on_t:
        ok &= np.abs(t - ot) >= gap_min
    cand = np.where(ok)[0]
    out = [0]
    for b in target[1:-1]:
        out.append(int(cand[np.argmin(np.abs(cand - b))]))
    out.append(n)
    return np.array(out)


# ---------------------------------------------------------------- models
GBM_PARAMS = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=60,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=5.0,
                  verbose=-1)


def fit_offset_gbm(X, y, offset, seed, n_jobs=3, **over):
    """GBM on residual hazard: boosting starts at the hazard log-odds."""
    p = {**GBM_PARAMS, **over}
    m = lgb.LGBMClassifier(random_state=seed, n_jobs=n_jobs, **p)
    m.fit(X, y, init_score=offset)
    return m


def predict_offset_gbm(m, X, offset):
    return sigmoid(np.asarray(offset, float) + m.predict(X, raw_score=True))


def fit_plain_gbm(X, y, seed, n_jobs=3, **over):
    p = {**GBM_PARAMS, **over}
    m = lgb.LGBMClassifier(random_state=seed, n_jobs=n_jobs, **p)
    m.fit(X, y)
    return m


# ---------------------------------------------------------------- evaluation
def subset_ap(y, score, next_is_first):
    """PR-AUC on all rows, and with positives restricted to rows whose next
    break is first-in-episode (resp. follow-on). Negatives are always all
    label-0 rows."""
    y = np.asarray(y)
    a_f = (y == 0) | next_is_first
    a_o = (y == 0) | ~next_is_first
    return {"all": float(average_precision_score(y, score)),
            "first": float(average_precision_score(y[a_f], score[a_f])),
            "follow": float(average_precision_score(y[a_o], score[a_o])),
            "prev_all": float(y.mean()), "prev_first": float(y[a_f].mean()),
            "prev_follow": float(y[a_o].mean())}


def budget_alarm(score, eligible, budget):
    """Alarm on for exactly the top round(budget * eligible) rows by score
    (label-free). Ties at the cut are broken by row order (earlier row first),
    so every score gets the same alarm-on time."""
    s = np.asarray(score, float)
    el = np.asarray(eligible, bool)
    out = np.zeros(len(s), bool)
    if el.sum() == 0:
        return out
    k = max(1, int(round(budget * el.sum())))
    idx = np.where(el)[0]
    order = np.lexsort((idx, -s[idx]))       # score descending, then row ascending
    out[idx[order[:k]]] = True
    return out


def warned_by(alarm, onsets, W, min_lead):
    """Per onset: did the alarm fire at some row t with min_lead <= o - t <= W."""
    alarm = np.asarray(alarm, bool)
    return np.array([bool(alarm[max(0, o - W):o - min_lead + 1].any()) for o in onsets])


def false_alarms(alarm, onsets, W):
    """Rising edges of the alarm with no break onset in (t, t+W]."""
    a = np.asarray(alarm, bool)
    starts = np.where(a & ~np.r_[False, a[:-1]])[0]
    on = np.sort(np.asarray(onsets))
    pos = np.searchsorted(on, starts, side="right")
    nxt = np.where(pos < len(on), on[np.minimum(pos, len(on) - 1)], 10 ** 12)
    return int(((nxt - starts) > W).sum()), int(len(starts))


# ---------------------------------------------------------------- bootstrap
def cluster_boot_diff(a, b, groups, B, rng, alpha=0.05):
    """Paired bootstrap of mean(a) - mean(b), resampling whole groups
    (episodes) with replacement. a, b: per-item 0/1 or float outcomes."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    g = np.asarray(groups)
    ug = np.unique(g)
    idx = [np.where(g == u)[0] for u in ug]
    d = np.empty(B)
    for i in range(B):
        pick = rng.integers(0, len(ug), len(ug))
        ii = np.concatenate([idx[j] for j in pick])
        d[i] = a[ii].mean() - b[ii].mean()
    return {"point": float(a.mean() - b.mean()), "lo": float(np.quantile(d, alpha / 2)),
            "hi": float(np.quantile(d, 1 - alpha / 2)), "share_above_0": float((d > 0).mean()), "B": B}


def day_block_ap_diff(y, s1, s2, blocks, B, rng, mask=None, alpha=0.05):
    """Paired bootstrap of AP(s1) - AP(s2), resampling whole blocks (days) of
    rows. `blocks` gives each row's block id; `mask` optionally restricts rows
    (e.g. first-break positives + all negatives)."""
    y = np.asarray(y)
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    bl = np.asarray(blocks)
    ub = np.unique(bl[m])
    rows = [np.where(m & (bl == u))[0] for u in ub]
    d = []
    for _ in range(B):
        ii = np.concatenate([rows[j] for j in rng.integers(0, len(ub), len(ub))])
        if y[ii].sum() == 0 or y[ii].sum() == len(ii):
            continue
        d.append(average_precision_score(y[ii], s1[ii]) - average_precision_score(y[ii], s2[ii]))
    d = np.array(d)
    return {"point": float(average_precision_score(y[m], s1[m]) - average_precision_score(y[m], s2[m])),
            "lo": float(np.quantile(d, alpha / 2)), "hi": float(np.quantile(d, 1 - alpha / 2)),
            "share_above_0": float((d > 0).mean()), "B": int(len(d))}
