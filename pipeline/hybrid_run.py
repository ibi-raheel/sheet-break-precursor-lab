"""Hazard + sensors: does a learned sensor model add anything the break-history
hazard cannot see? Forward-chaining, episode-safe folds; matched alarm-time
budgets; episode and day-block bootstrap CIs. Merges a `hybrid` section into
the results JSON written by run.py.

Usage: python -I pipeline/hybrid_run.py data/raw/processminer.csv src/data/results.json
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import features as fe
from pipeline import hybrid as hy
from pipeline import label as lb
from pipeline import load as ld
from pipeline import reliability as rl

SEED = 11
K_FOLDS = 5                  # contiguous blocks; blocks 2..5 are scored (forward chaining)
HORIZONS_MIN = [20, 60]      # 20 = primary (matches run.py); 60 = longer-horizon check
EPISODE_GAP_MIN = 120
BUDGETS = [0.02, 0.05, 0.10, 0.20]   # share of operating time the alarm is on
LEAD_MIN_ROWS = 5            # >= 10 min of warning
B_BOOT = 2000
DAY_ROWS = 720               # ~24 h at 2-minute sampling
SEEDS_ROBUST = [11, 12, 13]
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")

MODELS = ["hawkes", "sensor_gbm", "hybrid_gbm", "hybrid_logistic"]


def features(df, sig, src):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "features.pkl")
    key = (os.path.getsize(src), len(df))
    if os.path.exists(path):
        with open(path, "rb") as f:
            k, F = pickle.load(f)
        if k == key:
            return F
    F = fe.build(df, sig)
    with open(path, "wb") as f:
        pickle.dump((key, F), f)
    return F


def run_horizon(hm, X, Xf, label, elig, tmin, onsets, t_on, bounds, seed):
    n = len(label)
    h = hm // 2
    rows = np.arange(n)
    sc = {m: np.full(n, np.nan) for m in MODELS}
    fold_of = np.full(n, -1)
    fits = []
    for f in range(1, K_FOLDS):
        b0, b1 = bounds[f], bounds[f + 1]
        tr = rows < b0 - h            # purge: no training label can see into the test block
        te = (rows >= b0) & (rows < b1)
        fold_of[te] = f
        t_tr = t_on[onsets < b0]
        hk = rl.fit_hawkes(t_tr, tmin[b0])
        hz = rl.hawkes_p_next(tmin, t_on, hm, hk["mu"], hk["a"], hk["b"])   # past events only
        off = hy.logit(hz)
        mtr, mte = tr & elig, te & elig
        sc["hawkes"][mte] = hz[mte]
        g = hy.fit_plain_gbm(X[mtr], label[mtr], seed)
        sc["sensor_gbm"][mte] = g.predict_proba(X[mte])[:, 1]
        hg = hy.fit_offset_gbm(X[mtr], label[mtr], off[mtr], seed)
        sc["hybrid_gbm"][mte] = hy.predict_offset_gbm(hg, X[mte], off[mte])
        Xl = np.column_stack([Xf[mtr], off[mtr]])
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=4000))
        lr.fit(Xl, label[mtr])
        sc["hybrid_logistic"][mte] = lr.predict_proba(np.column_stack([Xf[mte], off[mte]]))[:, 1]
        fits.append({"fold": f, "train_rows": int(mtr.sum()), "test_rows": int(mte.sum()),
                     "train_breaks": int(len(t_tr)), "test_breaks": int(((onsets >= b0) & (onsets < b1)).sum()),
                     "hawkes": {k: hk[k] for k in ("mu", "a", "b", "branching")}})
    return sc, fold_of, fits


def evaluate(hm, sc, fold_of, label, elig, since_min, onsets, ep_id, ep_first, tmin, rng, boot=True):
    n = len(label)
    h = hm // 2
    test = (fold_of > 0) & elig
    y = label[test]
    nxt = onsets[np.minimum(np.searchsorted(onsets, np.where(test)[0], side="right"), len(onsets) - 1)]
    first_set = set(onsets[ep_first].tolist())
    nf = np.array([o in first_set for o in nxt])
    quiet = since_min[test] >= EPISODE_GAP_MIN
    out = {"h_min": hm, "test_rows": int(test.sum()), "positives": int(y.sum()), "ap": {}, "ap_quiet": {}}
    for m in MODELS:
        s = sc[m][test]
        out["ap"][m] = hy.subset_ap(y, s, nf)
        out["ap_quiet"][m] = float(hy.average_precision_score(y[quiet], s[quiet]))
    out["prev_quiet"] = float(y[quiet].mean())
    out["quiet_rows_share"] = float(quiet.mean())

    # test onsets whose warning window sits inside a scored block
    on_te = np.array([o for o in onsets if fold_of[o - h] > 0 and fold_of[o] == fold_of[o - h]])
    first_te = np.array([o in first_set for o in on_te])
    eps_te = ep_id[np.searchsorted(onsets, on_te)]
    shifts = test.sum() * 2 / 480
    out["test_breaks"] = int(len(on_te))
    out["test_first_breaks"] = int(first_te.sum())
    out["test_follow_breaks"] = int((~first_te).sum())
    out["test_shifts_operating"] = float(shifts)
    budgets = {}
    warned_all = {}
    for bgt in BUDGETS:
        row = {}
        for m in MODELS:
            alarm = np.zeros(n, bool)
            for f in np.unique(fold_of[fold_of > 0]):
                blk = (fold_of == f) & elig
                s = np.where(blk, sc[m], -np.inf)
                alarm |= hy.budget_alarm(s, blk, bgt)
            w = hy.warned_by(alarm, on_te, h, LEAD_MIN_ROWS)
            fa, edges = hy.false_alarms(alarm, on_te, h)
            warned_all[(bgt, m)] = w
            row[m] = {"on_share": float(alarm[test].mean()), "recall_all": float(w.mean()),
                      "recall_first": float(w[first_te].mean()), "recall_follow": float(w[~first_te].mean()),
                      "warned_first": int(w[first_te].sum()), "warned_follow": int(w[~first_te].sum()),
                      "fa_per_shift": float(fa / shifts), "alarms": edges}
        budgets[str(bgt)] = row
    out["budgets"] = budgets
    if not boot:
        return out

    # bootstrap: AP differences by ~1-day blocks; recall differences by episode
    blocks = (np.where(test)[0] // DAY_ROWS)
    a_f = (y == 0) | nf
    a_o = (y == 0) | ~nf
    ci = {}
    for a, b in (("hybrid_gbm", "hawkes"), ("hybrid_gbm", "sensor_gbm"), ("sensor_gbm", "hawkes"),
                 ("hybrid_logistic", "hawkes")):
        s1, s2 = sc[a][test], sc[b][test]
        ci[f"{a}_minus_{b}"] = {
            "ap_all": hy.day_block_ap_diff(y, s1, s2, blocks, B_BOOT, rng),
            "ap_first": hy.day_block_ap_diff(y, s1, s2, blocks, B_BOOT, rng, mask=a_f),
            "ap_follow": hy.day_block_ap_diff(y, s1, s2, blocks, B_BOOT, rng, mask=a_o),
            "ap_quiet": hy.day_block_ap_diff(y, s1, s2, blocks, B_BOOT, rng, mask=quiet)}
    out["ap_ci"] = ci
    rci = {}
    for bgt in BUDGETS:
        for a, b in (("hybrid_gbm", "hawkes"), ("hybrid_gbm", "sensor_gbm")):
            w1, w2 = warned_all[(bgt, a)], warned_all[(bgt, b)]
            rci[f"{bgt}:{a}_minus_{b}"] = {
                "first": hy.cluster_boot_diff(w1[first_te], w2[first_te], eps_te[first_te], B_BOOT, rng),
                "follow": hy.cluster_boot_diff(w1[~first_te], w2[~first_te], eps_te[~first_te], B_BOOT, rng),
                "all": hy.cluster_boot_diff(w1, w2, eps_te, B_BOOT, rng)}
    out["recall_ci"] = rci
    return out


def main(src, out_json):
    rng = np.random.default_rng(SEED)
    df = ld.load(src)
    sig = ld.signal_columns(df)
    n = len(df)
    onsets = ld.break_onsets(df["y"])
    tmin = (df["time"] - df["time"].iloc[0]).dt.total_seconds().values / 60
    t_on = tmin[onsets]
    ep_id, ep_first = rl.episodes(t_on, EPISODE_GAP_MIN)
    with open(out_json) as f:
        res = json.load(f)
    mask_k = int(res["meta"]["mask_rows"])     # measured in run.py on train weeks

    F = features(df, sig, src)
    Xcat = pd.get_dummies(df[ld.CATEGORICAL].astype(str), prefix="grade").astype(float)
    Xcat[ld.BINARY] = df[ld.BINARY].astype(float)
    X = pd.concat([F, Xcat], axis=1)
    Xf = X.fillna(0.0).values
    last = np.full(n, -1)
    for o in onsets:
        last[o:] = o
    since_min = np.where(last >= 0, tmin - tmin[np.maximum(last, 0)], 1e9)

    bounds = hy.episode_safe_bounds(n, onsets, tmin, K_FOLDS, EPISODE_GAP_MIN)
    print("fold bounds", bounds.tolist())
    hyb = {"folds": {"k": K_FOLDS, "bounds": bounds.tolist(),
                     "bounds_time": [df["time"].iloc[min(b, n - 1)].strftime("%Y-%m-%d %H:%M") for b in bounds]},
           "budgets": BUDGETS, "lead_min": LEAD_MIN_ROWS * 2, "gbm_params": hy.GBM_PARAMS,
           "models": {"hawkes": "self-exciting hazard, 3 parameters, break history only",
                      "sensor_gbm": "LightGBM on 531 sensor features + grade, no history",
                      "hybrid_gbm": "LightGBM boosted from the Hawkes log-odds (init_score): learns sensor corrections to the hazard",
                      "hybrid_logistic": "L2 logistic regression on sensor features + Hawkes log-odds"},
           "horizons": {}}
    for hm in HORIZONS_MIN:
        label, elig = lb.make_labels(n, onsets, hm // 2, mask_k)
        sc, fold_of, fits = run_horizon(hm, X, Xf, label, elig, tmin, onsets, t_on, bounds, SEED)
        ev = evaluate(hm, sc, fold_of, label, elig, since_min, onsets, ep_id, ep_first, tmin, rng)
        ev["fits"] = fits
        # seed robustness for the learned models (headline numbers only)
        rob = []
        for s in SEEDS_ROBUST[1:]:
            sc2, _, _ = run_horizon(hm, X, Xf, label, elig, tmin, onsets, t_on, bounds, s)
            e2 = evaluate(hm, sc2, fold_of, label, elig, since_min, onsets, ep_id, ep_first, tmin, rng, boot=False)
            rob.append({"seed": s, "ap_first_hybrid": e2["ap"]["hybrid_gbm"]["first"],
                        "ap_all_hybrid": e2["ap"]["hybrid_gbm"]["all"],
                        "recall_first_5pct_hybrid": e2["budgets"]["0.05"]["hybrid_gbm"]["recall_first"]})
        ev["seed_robustness"] = rob
        hyb["horizons"][str(hm)] = ev
        print(f"--- {hm} min ---")
        print("AP", json.dumps({m: {k: round(v, 4) for k, v in ev["ap"][m].items()} for m in MODELS}))
        print("AP quiet", {m: round(v, 4) for m, v in ev["ap_quiet"].items()}, "prev", round(ev["prev_quiet"], 4))
        for bgt in BUDGETS:
            print(bgt, {m: (r["recall_first"], r["recall_follow"], round(r["fa_per_shift"], 2), round(r["on_share"], 3))
                        for m, r in ev["budgets"][str(bgt)].items()})
        print("AP CI", json.dumps({k: {kk: (round(vv["point"], 4), round(vv["lo"], 4), round(vv["hi"], 4)) for kk, vv in v.items()} for k, v in ev["ap_ci"].items()}))
        print("recall CI", json.dumps({k: {kk: (round(vv["point"], 3), round(vv["lo"], 3), round(vv["hi"], 3)) for kk, vv in v.items()} for k, v in ev["recall_ci"].items()}))
        print("robust", rob)
    res["hybrid"] = hyb
    with open(out_json, "w") as f:
        json.dump(res, f, separators=(",", ":"))
    print("wrote", out_json)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
