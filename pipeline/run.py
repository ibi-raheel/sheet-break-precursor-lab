"""End-to-end pipeline: labels -> models across horizons -> TreeSHAP ->
candidate detectors -> lift + permutation null + BH -> families -> JSON.

Usage: python -I pipeline/run.py data/raw/processminer.csv src/data/results.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import features as fe
from pipeline import label as lb
from pipeline import load as ld
from pipeline import stats as st

SEED = 7
HORIZONS_MIN = [2, 4, 6, 10, 20, 30]
DET_MIN = 20                 # detector lookahead window (minutes)
B = 5000                     # permutation draws
ALPHA = 0.05
TEST_FRAC = 0.25
N_BLOCKS = 4                 # ~1 week each; detectors scored on blocks 2-4
RULES = {"level": "", "slope": "__slope5", "deviation": "__dev15"}


def zscore(df, cols, train_idx):
    mu = df.loc[train_idx, cols].mean()
    sd = df.loc[train_idx, cols].std().replace(0, 1)
    return ((df[cols] - mu) / sd).values


def fit_models(F, Xcat, label, eligible, tr, te, seed=SEED):
    X = pd.concat([F, Xcat], axis=1)
    mtr = tr & eligible
    mte = te & eligible
    ytr, yte = label[mtr], label[mte]
    spw = (len(ytr) - ytr.sum()) / max(1, ytr.sum())
    gbm = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=15,
                             min_child_samples=25, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, scale_pos_weight=spw, reg_lambda=1.0,
                             n_jobs=2, random_state=seed, verbose=-1)
    gbm.fit(X[mtr], ytr)
    p_gbm = gbm.predict_proba(X[mte])[:, 1]
    Xf = X.fillna(0.0)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000))
    lr.fit(Xf[mtr], ytr)
    p_lr = lr.predict_proba(Xf[mte])[:, 1]
    return gbm, X, mte, yte, p_gbm, p_lr


def score_detectors(df, F, sig, onsets, label_w, elig_w, W, bounds, eval_blocks, rng):
    """Threshold/direction for each block come only from rows before that block
    (rolling origin). Lift and the permutation null are computed on the
    evaluation blocks, pooling breaks; each break's null window is drawn from
    nominal windows in its own block."""
    n = len(df)
    lab_c = np.r_[0, np.cumsum(label_w)]
    inel_c = np.r_[0, np.cumsum(~elig_w)]
    ev_on, ev_blk, nom_by_block = [], [], {}
    for bb in eval_blocks:
        s0, s1 = bounds[bb], bounds[bb + 1]
        for o in onsets:
            if s0 + W <= o < s1 and elig_w[o - W:o].all():
                ev_on.append(o)
                ev_blk.append(bb)
        ends = np.arange(s0 + W, s1)
        ok = ((lab_c[ends + 1] - lab_c[ends + 1 - W]) == 0) & ((inel_c[ends + 1] - inel_c[ends + 1 - W]) == 0)
        nom_by_block[bb] = ends[ok]
    ev_on, ev_blk = np.array(ev_on), np.array(ev_blk)
    nom_ends = np.concatenate([nom_by_block[bb] for bb in eval_blocks])
    draws = np.empty((B, len(ev_on)), dtype=int)
    for j, bb in enumerate(ev_blk):
        pool = nom_by_block[bb]
        draws[:, j] = pool[rng.integers(0, len(pool), size=B)]
    t0 = bounds[eval_blocks[0]]
    days = sum((df["time"].iloc[bounds[bb + 1] - 1] - df["time"].iloc[bounds[bb]]).total_seconds()
               for bb in eval_blocks) / 86400
    near = np.zeros(n, bool)
    for o in ev_on:
        near[o - W:o] = True
    dets, fires = [], {}
    for s in sig:
        for rule, suf in RULES.items():
            v = F[s + suf].values
            fire = np.zeros(n, bool)
            info = None
            for bb in eval_blocks:
                past = np.arange(n) < bounds[bb]
                vn = v[past & elig_w & (label_w == 0)]
                vn = vn[~np.isnan(vn)]
                vp = v[past & elig_w & (label_w == 1)]
                vp = vp[~np.isnan(vp)]
                if len(vn) < 50 or np.std(vn) == 0 or len(vp) == 0:
                    info = None
                    break
                hi = np.median(vp) >= np.median(vn)
                thr = np.quantile(vn, 0.99 if hi else 0.01)
                seg = slice(bounds[bb], bounds[bb + 1])
                with np.errstate(invalid="ignore"):
                    fire[seg] = (v[seg] > thr) if hi else (v[seg] < thr)
                info = (hi, thr)
            if info is None:
                continue
            fire &= elig_w
            wa = st.window_any(fire, W)
            obs = wa[ev_on - 1].mean()
            nom_rate = wa[nom_ends].mean()
            null = wa[draws].mean(1)
            rise = np.where(fire & ~np.r_[False, fire[:-1]])[0]
            rise = rise[rise >= t0]
            tp = int(near[rise].sum())
            dets.append({"id": f"{s}:{rule}", "signal": s, "rule": rule,
                         "direction": "above" if info[0] else "below", "threshold": float(info[1]),
                         "coverage": float(obs), "nominal_rate": float(nom_rate),
                         "lift": float(obs / nom_rate) if nom_rate > 0 else None,
                         "precision": float(tp / len(rise)) if len(rise) else 0.0,
                         "false_alarms_per_day": float((len(rise) - tp) / days),
                         "p": float(st.perm_pvalue(obs, null)), "null": null})
            fires[f"{s}:{rule}"] = fire
    q = st.bh_adjust([d["p"] for d in dets])
    edges = np.linspace(0, 1, 26)
    for d, qq in zip(dets, q):
        d["q"] = float(qq)
        d["survives"] = bool(qq < ALPHA and (d["lift"] or 0) > 1)
        d["null_mean"] = float(d["null"].mean())
        d["null_hist"] = np.histogram(d.pop("null"), bins=edges)[0].tolist()
    dets.sort(key=lambda d: (d["q"], -(d["lift"] or 0)))
    ev = {"onsets": ev_on, "blocks": ev_blk, "nom_ends": nom_ends, "nom_by_block": nom_by_block,
          "draws": draws, "days": days}
    return {"dets": dets, "fires": fires, "ev": ev, "n_surv": int(sum(d["survives"] for d in dets))}


def main(src, out):
    rng = np.random.default_rng(SEED)
    df = ld.load(src)
    sig = ld.signal_columns(df)
    n = len(df)
    onsets = ld.break_onsets(df["y"])
    gaps = ld.sampling_gaps(df)
    split = int(n * (1 - TEST_FRAC))
    tr = np.arange(n) < split
    te = ~tr
    Z = zscore(df, sig, df.index[tr])
    mask_k, rec_lens = lb.restart_mask_len(Z, onsets[onsets < split])
    print("restart mask rows:", mask_k)

    F = fe.build(df, sig)
    Xcat = pd.get_dummies(df[ld.CATEGORICAL].astype(str), prefix="grade").astype(float)
    Xcat[ld.BINARY] = df[ld.BINARY].astype(float)
    # rows since last break onset, known at time t (used by the persistence baseline only)
    last = np.full(n, -10**6)
    for o in onsets:
        last[o:] = o
    since = np.arange(n) - last

    horizons = []
    shap_by_signal = None
    leak = {}
    for hm in HORIZONS_MIN:
        h = hm // 2
        label, elig = lb.make_labels(n, onsets, h, mask_k)
        gbm, X, mte, yte, p_gbm, p_lr = fit_models(F, Xcat, label, elig, tr, te)
        row = {"h_min": hm, "gbm": round(float(average_precision_score(yte, p_gbm)), 4),
               "logreg": round(float(average_precision_score(yte, p_lr)), 4),
               "prevalence": round(float(yte.mean()), 4),
               "persistence": round(float(average_precision_score(yte, -since[mte])), 4),
               "n_pos_test": int(yte.sum()), "n_test": int(len(yte))}
        horizons.append(row)
        print(row)
        if hm == DET_MIN:
            contrib = gbm.booster_.predict(X[mte], pred_contrib=True)[:, :-1]
            imp = pd.Series(np.abs(contrib).mean(0), index=X.columns)
            imp = imp.groupby([c.split("__")[0] for c in imp.index]).sum()
            shap_by_signal = imp.reindex(sig).fillna(0.0)
            labs, el2 = lb.make_labels(n, onsets, h, mask_k, extra_shift=1)
            _, _, m2, y2, pg2, _ = fit_models(F, Xcat, labs, el2, tr, te)
            leak = {"h_min": hm, "pr_auc": row["gbm"],
                    "pr_auc_shifted": round(float(average_precision_score(y2, pg2)), 4)}
            print("leakage check", leak)

    # ---------- detectors: rolling-origin evaluation ----------
    W = DET_MIN // 2
    label_w, elig_w = lb.make_labels(n, onsets, W, mask_k)
    bounds = np.linspace(0, n, N_BLOCKS + 1).astype(int)
    primary = score_detectors(df, F, sig, onsets, label_w, elig_w, W, bounds, list(range(1, N_BLOCKS)), rng)
    holdout = score_detectors(df, F, sig, onsets, label_w, elig_w, W, bounds, [N_BLOCKS - 1], rng)
    dets, fires, ev = primary["dets"], primary["fires"], primary["ev"]
    surv = [d for d in dets if d["survives"]]
    print("detectors:", len(dets), "survivors:", len(surv), "holdout-only survivors:", holdout["n_surv"])
    test_on, nom_ends, test_days, draws = ev["onsets"], ev["nom_ends"], ev["days"], ev["draws"]

    # ---------- null calibration: fake break times ----------
    calib_surv, calib_p = [], []
    for k in range(20):
        # fake onsets: one random nominal window end per real break, from that break's own block
        fake = np.array([ev["nom_by_block"][bb][rng.integers(len(ev["nom_by_block"][bb]))] for bb in ev["blocks"]]) + 1
        ps = []
        for d in dets:
            wa = st.window_any(fires[d["id"]], W)
            obs = wa[fake - 1].mean()
            null = wa[draws[:1000]].mean(1)
            ps.append(st.perm_pvalue(obs, null))
        qs = st.bh_adjust(ps)
        calib_surv.append(int((qs < ALPHA).sum()))
        calib_p.extend(ps)
    calib = {"runs": 20, "survivors_per_run": calib_surv,
             "mean_survivors": float(np.mean(calib_surv)),
             "p_hist": np.histogram(calib_p, bins=np.linspace(0, 1, 11))[0].tolist(),
             "share_p_below_005": float(np.mean(np.array(calib_p) < 0.05))}
    print("calibration", calib["mean_survivors"], calib["share_p_below_005"])

    near = np.zeros(n, bool)
    for o in test_on:
        near[o - W:o] = True
    if surv:
        union = np.zeros(n, bool)
        for d in surv:
            union |= fires[d["id"]]
        wa = st.window_any(union, W)
        rise = np.where(union & ~np.r_[False, union[:-1]])[0]
        rise = rise[rise >= bounds[1]]
        union_stats = {"coverage": float(wa[test_on - 1].mean()),
                       "false_alarms_per_day": float((~near[rise]).sum() / test_days),
                       "n_detectors": len(surv)}
    else:
        union_stats = {"coverage": 0.0, "false_alarms_per_day": 0.0, "n_detectors": 0}

    # ---------- replay ----------
    top = surv[:4] if surv else dets[:4]
    replay = []
    for o in test_on:
        lo, hi_ = o - 30, min(n, o + 4)
        tracks = []
        for d in top:
            col = d["signal"] + RULES[d["rule"]]
            v = F[col].values[lo:hi_]
            f = fires[d["id"]][lo:hi_]
            first = np.where(fires[d["id"]][o - W:o])[0]
            tracks.append({"id": d["id"], "values": [None if np.isnan(x) else round(float(x), 4) for x in v],
                           "fire": f.astype(int).tolist(), "threshold": d["threshold"],
                           "lead_min": int((W - first[0]) * 2) if len(first) else None})
        replay.append({"onset": int(o), "time": df["time"].iloc[o].strftime("%Y-%m-%d %H:%M"),
                       "minutes": [int(2 * (i - o)) for i in range(lo, hi_)], "tracks": tracks})

    # ---------- families + SHAP vs lift ----------
    rho = spearmanr(df.loc[tr, sig].values).correlation
    rho = np.nan_to_num(np.abs(rho), nan=0.0)
    np.fill_diagonal(rho, 1.0)
    dist = squareform(1 - rho, checks=False)
    fam = fcluster(linkage(dist, "average"), t=0.3, criterion="distance")
    shap_rank = shap_by_signal.rank(ascending=False, method="first").astype(int)
    scatter = []
    for i, s in enumerate(sig):
        ds = [d for d in dets if d["signal"] == s and d["lift"] is not None]
        best = max(ds, key=lambda d: d["lift"]) if ds else None
        scatter.append({"signal": s, "shap": float(shap_by_signal[s]), "shap_rank": int(shap_rank[s]),
                        "best_lift": best["lift"] if best else None, "best_rule": best["rule"] if best else None,
                        "best_q": min(d["q"] for d in ds) if ds else None,
                        "survives": any(d["survives"] for d in ds), "family": int(fam[i])})
    sl = [(p["shap"], p["best_lift"]) for p in scatter if p["best_lift"] is not None]
    rho_sl = float(spearmanr([a for a, _ in sl], [b for _, b in sl]).correlation)
    top10 = sorted(scatter, key=lambda p: p["shap_rank"])[:10]

    # timeline trace: the top surviving detector's raw signal
    trace_sig = top[0]["signal"]
    masks = [[int(o), int(min(n - 1, o + mask_k))] for o in onsets]
    res = {
        "meta": {"rows": n, "signals": len(sig), "breaks": int(len(onsets)),
                 "positive_rows": int(df["y"].sum()), "break_row_share": float(df["y"].mean()),
                 "start": df["time"].iloc[0].strftime("%Y-%m-%d %H:%M"),
                 "end": df["time"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                 "sampling_min": 2, "gaps": int(len(gaps)), "mask_rows": mask_k,
                 "recovery_rows_median": float(np.median(rec_lens)),
                 "split_row": split, "split_time": df["time"].iloc[split].strftime("%Y-%m-%d %H:%M"),
                 "eval_breaks": int(len(test_on)), "eval_days": round(test_days, 2), "holdout_survivors": holdout["n_surv"], "holdout_breaks": int(len(holdout["ev"]["onsets"])), "blocks": bounds.tolist(),
                 "det_window_min": DET_MIN, "permutations": B, "alpha": ALPHA,
                 "nominal_windows": int(len(nom_ends))},
        "timeline": {"signal": trace_sig, "minutes": ((df["time"] - df["time"].iloc[0]).dt.total_seconds() / 60).astype(int).tolist(),
                     "values": [round(float(x), 3) for x in df[trace_sig].values],
                     "onsets": onsets.tolist(), "masks": masks},
        "horizons": horizons,
        "detectors": dets,
        "union": union_stats,
        "replay": replay,
        "scatter": scatter,
        "shap_lift_spearman": rho_sl,
        "top10_shap_survivors": int(sum(p["survives"] for p in top10)),
        "validation": {"leakage": leak, "null_calibration": calib},
    }
    with open(out, "w") as f:
        json.dump(res, f, separators=(",", ":"))
    print("wrote", out, "survivors", len(surv), "union", union_stats, "rho shap-lift", rho_sl)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
