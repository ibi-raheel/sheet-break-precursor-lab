"""Robustness checks on the 20-minute first-break claim (adversarial review).

1. Lead-time check. The 20-minute label window includes the last 2 to 8
   minutes before a break, which an operator cannot act on. Re-score
   first-break PR-AUC with only positives at least `k` rows (2k minutes) ahead
   and re-run the per-block rotation null on that subset.
2. Grade-only check. Product grade (one-hot x28) plus the binary x61 with the
   same LightGBM settings: does the "sensor" model's first-break skill come
   from grade alone?

Same folds, labels and seed as hybrid_run.py. Merges `hybrid.lead_check` into
the results JSON.

Usage: python -I pipeline/lead_check.py data/raw/processminer.csv src/data/results.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from pipeline import hybrid as hy
from pipeline import hybrid_run as hr
from pipeline import label as lb
from pipeline import load as ld
from pipeline import reliability as rl

HM = 20
LEADS = [1, 3, 5]            # rows: >= 2, 6, 10 minutes ahead
R = 1000
MODELS = ["hawkes", "sensor_gbm", "hybrid_gbm", "hybrid_superpose", "grade_only"]


def main(src, out_json):
    df = ld.load(src)
    sig = ld.signal_columns(df)
    n = len(df)
    onsets = ld.break_onsets(df["y"])
    tmin = (df["time"] - df["time"].iloc[0]).dt.total_seconds().values / 60
    t_on = tmin[onsets]
    ep_id, ep_first = rl.episodes(t_on, hr.EPISODE_GAP_MIN)
    with open(out_json) as f:
        res = json.load(f)
    mask_k = int(res["meta"]["mask_rows"])
    F = hr.features(df, sig, src)
    Xcat = pd.get_dummies(df[ld.CATEGORICAL].astype(str), prefix="grade").astype(float)
    Xcat[ld.BINARY] = df[ld.BINARY].astype(float)
    X = pd.concat([F, Xcat], axis=1)
    Xf = X.fillna(0.0).values
    last = np.full(n, -1)
    for o in onsets:
        last[o:] = o
    since = np.where(last >= 0, tmin - tmin[np.maximum(last, 0)], 1e9)
    bounds = hy.episode_safe_bounds(n, onsets, tmin, hr.K_FOLDS, hr.EPISODE_GAP_MIN)
    h = HM // 2
    label, elig = lb.make_labels(n, onsets, h, mask_k)
    sc, fold_of, _ = hr.run_horizon(HM, X, Xf, label, elig, tmin, onsets, t_on, bounds, hr.SEED, since)

    rows = np.arange(n)
    Xg = Xcat.values
    sg = np.full(n, np.nan)
    for f in range(1, hr.K_FOLDS):
        b0, b1 = bounds[f], bounds[f + 1]
        tr = (rows < b0 - h) & elig
        te = (rows >= b0) & (rows < b1) & elig
        g = hy.fit_plain_gbm(Xg[tr], label[tr], hr.SEED)
        sg[te] = g.predict_proba(Xg[te])[:, 1]
    sc["grade_only"] = sg

    test = (fold_of > 0) & elig
    ti = np.where(test)[0]
    y = label[ti]
    nxt = onsets[np.minimum(np.searchsorted(onsets, ti, side="right"), len(onsets) - 1)]
    lead = nxt - ti
    first_set = set(onsets[ep_first].tolist())
    nf = np.array([o in first_set for o in nxt])
    blk_idx = [np.where((fold_of == f) & elig)[0] for f in np.unique(fold_of[fold_of > 0])]
    yfull = np.zeros(n, int)
    yfull[ti] = y
    nrng = np.random.default_rng(hr.SEED + 2000)
    out = {"h_min": HM, "R": R, "leads_rows": LEADS,
           "method": "first-break PR-AUC with positives restricted to rows >= k rows before the break "
                     "(negatives = all label-0 test rows); per-block rotation null as in ap_chance",
           "models": {}}
    for m in MODELS:
        s_full = np.where(test, sc[m], 0.0)
        row = {}
        for k in LEADS:
            keep = (y == 0) | (nf & (lead >= k))
            msk = np.zeros(n, bool)
            msk[ti] = keep
            obs = float(ap(y[keep], sc[m][ti][keep]))
            nl = hy.shift_null_ap(yfull, s_full, blk_idx, msk, R, nrng)
            row[str(k)] = {"obs": obs, "prev": float(y[keep].mean()), "null_mean": float(nl.mean()),
                           "p": float((1 + (nl >= obs).sum()) / (1 + R))}
        out["models"][m] = row
    rng = np.random.default_rng(hr.SEED + 3000)
    blocks = ti // hr.DAY_ROWS
    a_f = (y == 0) | nf
    out["sensor_minus_grade_first"] = hy.day_block_ap_diff(y, sc["sensor_gbm"][ti], sc["grade_only"][ti],
                                                           blocks, hr.B_BOOT, rng, mask=a_f)
    print(json.dumps(out, indent=1))
    res.setdefault("hybrid", {})["lead_check"] = out
    with open(out_json, "w") as f:
        json.dump(res, f, separators=(",", ":"))
    print("wrote", out_json)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
