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
from pipeline import reliability as rl

SEED = 7
HORIZONS_MIN = [2, 4, 6, 10, 20, 30]
DET_MIN = 20                 # detector lookahead window (minutes)
B = 5000                     # permutation draws
ALPHA = 0.05
TEST_FRAC = 0.25
N_BLOCKS = 4                 # ~1 week each; detectors scored on blocks 2-4
EPISODE_GAP_MIN = 120       # breaks closer than this are one episode
B_BOOT = 2000
BOOT_BLOCK = 360             # rows (12 h) per moving block
LEAD_MIN_ROWS = 5            # >= 10 min of warning
CHECK_MIN = 5.0              # assumed operator check per alarm (minutes)
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


def gbm_fit_predict(X, y, mtr, mpred, seed=SEED):
    ytr = y[mtr]
    spw = (len(ytr) - ytr.sum()) / max(1, ytr.sum())
    m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=15,
                           min_child_samples=25, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, scale_pos_weight=spw, reg_lambda=1.0,
                           n_jobs=2, random_state=seed, verbose=-1)
    m.fit(X[mtr], ytr)
    return m.predict_proba(X[mpred])[:, 1]


def oof_train_scores(X, y, elig, tr, n_folds=3):
    """Blocked K-fold scores for training rows (each fold scored by a model
    that never saw it). Used only to set alarm thresholds on train weeks."""
    out = np.full(len(y), np.nan)
    idx = np.where(tr)[0]
    edges = np.linspace(idx[0], idx[-1] + 1, n_folds + 1).astype(int)
    rows = np.arange(len(y))
    for f in range(n_folds):
        fold = (rows >= edges[f]) & (rows < edges[f + 1])
        out[fold & elig] = gbm_fit_predict(X, y, tr & ~fold & elig, fold & elig)
    return out


def score_detectors(df, F, sig, onsets, label_w, elig_w, W, bounds, eval_blocks, rng, first_set=None):
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
    # episode-level: one window per episode, ending at its first break
    first_set = first_set if first_set is not None else set(onsets.tolist())
    ep_sel = np.array([o in first_set for o in ev_on], bool)
    ep_on, ep_blk = ev_on[ep_sel], ev_blk[ep_sel]
    draws_ep = np.empty((B, len(ep_on)), dtype=int)
    for j, bb in enumerate(ep_blk):
        pool = nom_by_block[bb]
        draws_ep[:, j] = pool[rng.integers(0, len(pool), size=B)]
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
                         "p": float(st.perm_pvalue(obs, null)), "null": null,
                         "coverage_ep": float(wa[ep_on - 1].mean()),
                         "covered_ep": int(wa[ep_on - 1].sum()),
                         "p_ep": float(st.perm_pvalue(wa[ep_on - 1].mean(), wa[draws_ep].mean(1)))})
            fires[f"{s}:{rule}"] = fire
    q = st.bh_adjust([d["p"] for d in dets])
    q_ep = st.bh_adjust([d["p_ep"] for d in dets])
    for d, qq in zip(dets, q_ep):
        d["q_ep"] = float(qq)
        d["survives_ep"] = bool(qq < ALPHA and (d["lift"] or 0) > 1)
    edges = np.linspace(0, 1, 26)
    for d, qq in zip(dets, q):
        d["q"] = float(qq)
        d["survives"] = bool(qq < ALPHA and (d["lift"] or 0) > 1)
        d["null_mean"] = float(d["null"].mean())
        d["null_hist"] = np.histogram(d.pop("null"), bins=edges)[0].tolist()
    dets.sort(key=lambda d: (d["q"], -(d["lift"] or 0)))
    ev = {"onsets": ev_on, "blocks": ev_blk, "nom_ends": nom_ends, "nom_by_block": nom_by_block,
          "draws": draws, "days": days, "ep_on": ep_on, "ep_blk": ep_blk, "draws_ep": draws_ep}
    return {"dets": dets, "fires": fires, "ev": ev, "n_surv": int(sum(d["survives"] for d in dets)),
            "n_surv_ep": int(sum(d["survives_ep"] for d in dets))}


def main(src, out):
    rng = np.random.default_rng(SEED)
    df = ld.load(src)
    sig = ld.signal_columns(df)
    n = len(df)
    onsets = ld.break_onsets(df["y"])
    gaps = ld.sampling_gaps(df)
    tmin = (df["time"] - df["time"].iloc[0]).dt.total_seconds().values / 60
    t_on = tmin[onsets]
    ep_id, ep_first = rl.episodes(t_on, EPISODE_GAP_MIN)
    first_set = set(onsets[ep_first].tolist())
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

    # ---------- reliability: inter-break times on train weeks only ----------
    T_split = tmin[split]
    t_tr = t_on[onsets < split]
    tau = np.diff(t_tr)
    cens = T_split - t_tr[-1]
    m_exp = rl.fit_exponential(tau, cens)
    m_wb = rl.fit_weibull(tau, cens)
    m_hk = rl.fit_hawkes(t_tr, T_split)
    from scipy.stats import kstest
    ks = kstest(tau, lambda x: rl.weibull_cdf(x, m_wb["k"], m_wb["lam"]))
    ks_exp = kstest(tau, lambda x: 1 - np.exp(-x / m_exp["lam"]))
    brng = np.random.default_rng(SEED + 1)
    kb = []
    for _ in range(1000):
        tb = tau[brng.integers(0, len(tau), len(tau))]
        kb.append(rl.fit_weibull(tb)["k"])
    # operational rule (train): P(next break within 2 h of a break) vs base rate
    nxt_gap = np.diff(t_tr)
    p_follow = float((nxt_gap < EPISODE_GAP_MIN).mean())
    rate_tr = (len(t_tr) - 1) / (T_split - t_tr[0])
    p_base = float(1 - np.exp(-rate_tr * EPISODE_GAP_MIN))
    fol = nxt_gap[nxt_gap < EPISODE_GAP_MIN]
    watch = {f"q{int(qq*100)}": float(np.quantile(fol, qq)) for qq in (0.5, 0.8, 0.9)}
    since_min = np.where(last >= 0, tmin - tmin[np.maximum(last, 0)], tmin)
    reliability = {
        "train_breaks": int(len(t_tr)), "train_intervals": int(len(tau)),
        "exponential": {**m_exp, "aic": rl.aic(m_exp), "ks": float(ks_exp.statistic), "ks_p": float(ks_exp.pvalue)},
        "weibull": {**m_wb, "aic": rl.aic(m_wb), "ks": float(ks.statistic), "ks_p": float(ks.pvalue),
                    "k_ci": [float(np.quantile(kb, 0.025)), float(np.quantile(kb, 0.975))]},
        "hawkes": {**m_hk, "aic": rl.aic(m_hk), "excitation_halflife_min": float(np.log(2) / m_hk["b"])},
        "cv_intervals": float(tau.std() / tau.mean()),
        "share_gaps_lt_1h": float((tau < 60).mean()), "share_gaps_lt_2h": float((tau < 120).mean()),
        "p_break_within_2h_after_break": p_follow, "p_break_in_random_2h": p_base,
        "followon_since_prev_min": watch,
    }
    print("reliability", {k: reliability[k] for k in ("cv_intervals", "p_break_within_2h_after_break", "p_break_in_random_2h")},
          "k", m_wb["k"], reliability["weibull"]["k_ci"], "AIC exp/wb/hk", rl.aic(m_exp), rl.aic(m_wb), rl.aic(m_hk),
          "branching", m_hk["branching"])

    horizons = []
    shap_by_signal = None
    leak = {}
    keep = {}
    for hm in HORIZONS_MIN:
        h = hm // 2
        label, elig = lb.make_labels(n, onsets, h, mask_k)
        gbm, X, mte, yte, p_gbm, p_lr = fit_models(F, Xcat, label, elig, tr, te)
        hz_wb = rl.weibull_p_next(since_min, hm, m_wb["k"], m_wb["lam"])
        hz_hk = rl.hawkes_p_next(tmin, t_on, hm, m_hk["mu"], m_hk["a"], m_hk["b"])
        keep[hm] = {"yte": yte, "gbm": p_gbm, "pers": -since[mte].astype(float), "hawkes": hz_hk[mte],
                    "weibull": hz_wb[mte], "mte": mte, "label": label, "elig": elig, "X": X,
                    "hz_hk_all": hz_hk}
        row = {"h_min": hm, "gbm": round(float(average_precision_score(yte, p_gbm)), 4),
               "logreg": round(float(average_precision_score(yte, p_lr)), 4),
               "prevalence": round(float(yte.mean()), 4),
               "persistence": round(float(average_precision_score(yte, -since[mte])), 4),
               "hazard_weibull": round(float(average_precision_score(yte, hz_wb[mte])), 4),
               "hazard_hawkes": round(float(average_precision_score(yte, hz_hk[mte])), 4),
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
    primary = score_detectors(df, F, sig, onsets, label_w, elig_w, W, bounds, list(range(1, N_BLOCKS)), rng, first_set)
    holdout = score_detectors(df, F, sig, onsets, label_w, elig_w, W, bounds, [N_BLOCKS - 1], rng, first_set)
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

    # ---------- clustered null calibration: shift whole real episodes ----------
    ev_ep = ep_id[np.searchsorted(onsets, test_on)]
    groups = [np.where(ev_ep == e)[0] for e in np.unique(ev_ep)]
    wa_all = {d["id"]: st.window_any(fires[d["id"]], W) for d in dets}
    g_blk = np.array([ev["blocks"][g[0]] for g in groups])
    draws_g = np.empty((1000, len(groups)), dtype=int)
    for j, bb in enumerate(g_blk):
        pool = ev["nom_by_block"][bb]
        draws_g[:, j] = pool[rng.integers(0, len(pool), size=1000)]
    cl_ind, cl_ep, cl_p_ind, cl_p_ep = [], [], [], []
    for k in range(20):
        fake_all, fake_first = np.empty(len(test_on), int), np.empty(len(groups), int)
        for j, g in enumerate(groups):
            off = test_on[g] - test_on[g[0]]
            bb = g_blk[j]
            pool = ev["nom_by_block"][bb]
            pool = pool[pool + 1 + off.max() < bounds[bb + 1]]
            u = pool[rng.integers(len(pool))]
            fake_all[g] = u + 1 + off
            fake_first[j] = u + 1
        pi, pe = [], []
        for d in dets:
            wa = wa_all[d["id"]]
            pi.append(st.perm_pvalue(wa[fake_all - 1].mean(), wa[draws[:1000]].mean(1)))
            pe.append(st.perm_pvalue(wa[fake_first - 1].mean(), wa[draws_g].mean(1)))
        cl_ind.append(int((st.bh_adjust(pi) < ALPHA).sum()))
        cl_ep.append(int((st.bh_adjust(pe) < ALPHA).sum()))
        cl_p_ind.extend(pi)
        cl_p_ep.extend(pe)
    calib_cluster = {"runs": 20, "episodes": len(groups),
                     "independent_null": {"survivors_per_run": cl_ind, "mean_survivors": float(np.mean(cl_ind)),
                                          "runs_with_false_survivor": int(sum(c > 0 for c in cl_ind)),
                                          "share_p_below_005": float(np.mean(np.array(cl_p_ind) < 0.05))},
                     "episode_null": {"survivors_per_run": cl_ep, "mean_survivors": float(np.mean(cl_ep)),
                                      "runs_with_false_survivor": int(sum(c > 0 for c in cl_ep)),
                                      "share_p_below_005": float(np.mean(np.array(cl_p_ep) < 0.05))}}
    print("clustered calibration", calib_cluster)

    # ---------- episode-level detector result ----------
    ep_on = ev["ep_on"]
    surv_ep = [d for d in dets if d["survives_ep"]]
    top_det = dets[0]
    episode_level = {"eval_episodes": int(len(ep_on)), "eval_breaks": int(len(test_on)),
                     "eval_episode_groups": len(groups), "survivors": len(surv_ep),
                     "survivor_ids": [d["id"] for d in surv_ep],
                     "top_break_level": {k: top_det[k] for k in ("id", "coverage", "q", "coverage_ep", "covered_ep", "p_ep", "q_ep", "survives_ep")},
                     "best_episode_level": sorted(({k: d[k] for k in ("id", "lift", "coverage_ep", "covered_ep", "p_ep", "q_ep")} for d in dets),
                                                  key=lambda d: d["p_ep"])[:5]}
    print("episode level", episode_level)

    # ---------- bootstrap CIs ----------
    boot = {}
    for hm in (20, 30):
        kk = keep[hm]
        boot[f"gbm_minus_persistence_{hm}"] = rl.ap_diff_ci(kk["yte"], kk["gbm"], kk["pers"], BOOT_BLOCK, B_BOOT, brng)
        boot[f"gbm_minus_hawkes_{hm}"] = rl.ap_diff_ci(kk["yte"], kk["gbm"], kk["hawkes"], BOOT_BLOCK, B_BOOT, brng)
    # lift CI for the top detector: resample episodes for P_pre, moving blocks of nominal windows for P_nom
    wa0 = wa_all[top_det["id"]]
    hit = wa0[test_on - 1].astype(float)
    nom_hits = wa0[nom_ends].astype(float)
    lifts = []
    for _ in range(B_BOOT):
        gs = [groups[i] for i in brng.integers(0, len(groups), len(groups))]
        ppre = np.concatenate([hit[g] for g in gs]).mean()
        pn = nom_hits[rl.mbb_indices(len(nom_hits), BOOT_BLOCK, brng)].mean()
        if pn > 0:
            lifts.append(ppre / pn)
    boot["top_detector_lift"] = {"id": top_det["id"], "point": top_det["lift"],
                                 "lo": float(np.quantile(lifts, 0.025)), "hi": float(np.quantile(lifts, 0.975)),
                                 "share_le_1": float(np.mean(np.array(lifts) <= 1))}
    print("bootstrap", boot)

    # ---------- 20-min: time-since-break feature, first vs follow-on, event-level alarms ----------
    k20 = keep[20]
    X20, lab20, el20, mte20 = k20["X"], k20["label"], k20["elig"], k20["mte"]
    Xs = X20.copy()
    Xs["rows_since_break"] = np.minimum(since, 720).astype(float)
    p_gs = gbm_fit_predict(Xs, lab20, tr & el20, mte20)
    rows_te = np.where(mte20)[0]
    nxt = onsets[np.minimum(np.searchsorted(onsets, rows_te, side="right"), len(onsets) - 1)]
    is_first_next = np.array([o in first_set for o in nxt])
    y = k20["yte"]
    scores20 = {"gbm": k20["gbm"], "gbm_since": p_gs, "hawkes": k20["hawkes"], "persistence": k20["pers"]}
    split_ap = {}
    for name, sc in scores20.items():
        a_f = (y == 0) | is_first_next
        a_o = (y == 0) | ~is_first_next
        split_ap[name] = {"all": float(average_precision_score(y, sc)),
                          "first_in_episode": float(average_precision_score(y[a_f], sc[a_f])),
                          "follow_on": float(average_precision_score(y[a_o], sc[a_o])),
                          "prev_first": float(y[a_f].mean()), "prev_follow": float(y[a_o].mean())}
    print("first vs follow-on PR-AUC", split_ap)

    # full-length score arrays: out-of-fold on train rows, model scores on test rows
    def full(oof, test_scores):
        a = np.where(np.isnan(oof), -np.inf, oof)
        a[mte20] = test_scores
        return a
    oof_g = oof_train_scores(X20, lab20, el20, tr)
    oof_gs = oof_train_scores(Xs, lab20, el20, tr)
    alarm_scores = {"gbm": full(oof_g, k20["gbm"]), "gbm_since": full(oof_gs, p_gs),
                    "hawkes": np.where(el20, k20["hz_hk_all"], -np.inf)}
    on_tr = onsets[onsets < split]
    on_te = onsets[onsets >= split + W]
    first_te = np.array([o in first_set for o in on_te])
    shifts_tr = (tr & el20).sum() * 2 / 480
    shifts_te = (te & el20).sum() * 2 / 480
    te_days = (tmin[-1] - tmin[split]) / 1440
    events = {"test_breaks": int(len(on_te)), "test_episodes": int(first_te.sum()),
              "test_days": float(te_days), "test_shifts_operating": float(shifts_te), "targets": {}}
    for target in (0.5, 1.0, 2.0):
        rows = {}
        for name, sc in alarm_scores.items():
            thr = rl.threshold_for_fa(np.where(tr, sc, -np.inf), tr & el20, on_tr, W, shifts_tr, target)
            above = (sc > thr) & te & el20
            starts, true = rl.alarm_events(above, on_te, W)
            wflag, lead = rl.warned(above, on_te, W, LEAD_MIN_ROWS)
            lw = lead[wflag] * 2
            rows[name] = {"threshold": thr, "recall_breaks": float(wflag.mean()),
                          "warned_breaks": int(wflag.sum()),
                          "recall_first": float(wflag[first_te].mean()),
                          "recall_follow": float(wflag[~first_te].mean()) if (~first_te).any() else None,
                          "fa_per_shift_test": float((len(starts) - true.sum()) / shifts_te),
                          "on_share_test": float(above.sum() / (te & el20).sum()),
                          "lead_min_median": float(np.median(lw)) if len(lw) else None}
        # the surviving single-sensor rule at its fixed threshold (no tuning)
        x6 = fires[top_det["id"]] & te & el20
        st_, tr_ = rl.alarm_events(x6, on_te, W)
        wflag, lead = rl.warned(x6, on_te, W, LEAD_MIN_ROWS)
        rows[top_det["id"] + " (fixed)"] = {"recall_breaks": float(wflag.mean()), "warned_breaks": int(wflag.sum()),
                                           "recall_first": float(wflag[first_te].mean()),
                                           "fa_per_shift_test": float((len(st_) - tr_.sum()) / shifts_te),
                                           "on_share_test": float(x6.sum() / (te & el20).sum()),
                                           "lead_min_median": float(np.median(lead[wflag] * 2)) if wflag.any() else None}
        events["targets"][str(target)] = rows
    print("event-level", json.dumps(events["targets"]["1.0"]))

    # ---------- measured downtime and break-even ----------
    dt = rl.post_break_downtime(tmin, onsets)
    gap_rows = ld.sampling_gaps(df)
    gaps_after_break = int(sum(np.any((onsets < g) & (g <= onsets + 2)) for g in gap_rows))
    span_h = tmin[-1] / 60
    down = {"breaks_with_gap": int((dt > 0).sum()), "gaps_total": int(len(gap_rows)),
            "gaps_within_2_rows_after_break": gaps_after_break,
            "median_min_when_gap": float(np.median(dt[dt > 0])),
            "iqr_min_when_gap": [float(np.quantile(dt[dt > 0], 0.25)), float(np.quantile(dt[dt > 0], 0.75))],
            "mean_min_per_break": float(dt.mean()), "total_hours": float(dt.sum() / 60),
            "hours_per_day": float(dt.sum() / 60 / (span_h / 24)), "share_of_calendar": float(dt.sum() / 60 / span_h),
            "first_in_episode_mean_min": float(dt[ep_first].mean()), "follow_on_mean_min": float(dt[~ep_first].mean())}
    print("downtime", down)
    bpd = len(on_te) / te_days
    econ = {}
    for name, r in events["targets"]["1.0"].items():
        econ[name] = rl.breakeven(r["recall_breaks"], bpd, down["mean_min_per_break"],
                                  r["fa_per_shift_test"] * 3, CHECK_MIN)
    econ_assump = {"breaks_per_day_test": float(bpd), "downtime_min_per_break": down["mean_min_per_break"],
                   "check_min_per_false_alarm": CHECK_MIN, "fa_target_per_shift": 1.0,
                   "note": "e = share of warned breaks an operator prevents; not measurable from this data"}
    print("breakeven", econ)

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
        "validation": {"leakage": leak, "null_calibration": calib, "clustered_null_calibration": calib_cluster},
        "episodes": {"gap_min": EPISODE_GAP_MIN, "all_breaks": int(len(onsets)), "all_episodes": int(ep_first.sum()),
                     "first_in_episode": int(ep_first.sum()), "follow_on": int((~ep_first).sum())},
        "episode_level": episode_level,
        "bootstrap": boot,
        "reliability": reliability,
        "first_vs_follow": split_ap,
        "events": events,
        "downtime": down,
        "economics": {"assumptions": econ_assump, "at_1_fa_per_shift": econ},
    }
    with open(out, "w") as f:
        json.dump(res, f, separators=(",", ":"))
    print("wrote", out, "survivors", len(surv), "union", union_stats, "rho shap-lift", rho_sl)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
