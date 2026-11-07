"""First-week triage for a line with rare, clustered failures.

A thin CLI over the existing pipeline: no new statistics. It sizes the
downtime, tests whether events cluster, routes every event (follow-on ->
hazard / watch window, first-in-episode -> sensor screening) and labels every
candidate detector as holding per episode, a one-episode artefact, or noise.

Usage: python -I pipeline/triage.py line.csv [--model] [--out triage.json]
Input: columns `time`, `y` (0/1 event flag per row), `x1..xn` sensors; the
grade / categorical column is set with --grade (default x28).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline import hybrid as hy
from pipeline import hybrid_run as hr
from pipeline import label as lb
from pipeline import load as ld
from pipeline import reliability as rl
from pipeline import run

HOLDS, ARTEFACT, NOISE = "HOLDS PER EPISODE", "ONE-EPISODE ARTEFACT", "NOISE"
ACTION = {HOLDS: "take to process engineer", ARTEFACT: "drop", NOISE: "drop"}


def load_csv(path):
    try:
        return ld.load(path)
    except ValueError:                      # other timestamp formats
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        return df.sort_values("time").reset_index(drop=True)


def rule_label(d):
    if d["survives_ep"]:
        return HOLDS
    return ARTEFACT if d["survives"] else NOISE


def cost_card(df, tmin, onsets):
    dt = rl.post_break_downtime(tmin, onsets)
    span_d = tmin[-1] / 1440
    return {"events": int(len(onsets)), "sampling_gaps": int(len(ld.sampling_gaps(df))),
            "events_with_gap": int((dt > 0).sum()), "downtime_min_per_event": float(dt.mean()),
            "downtime_hours": float(dt.sum() / 60), "line_hours_per_day": float(dt.sum() / 60 / span_d),
            "days": float(span_d), "note": "inferred from timestamp gaps"}


def clustering_card(t_on, T_split, gap_min, seed):
    """Fits on the training share of the record, as pipeline/run.py does."""
    t_tr = t_on[t_on < T_split]
    tau = np.diff(t_tr)
    cens = T_split - t_tr[-1]
    m_exp, m_wb, m_hk = rl.fit_exponential(tau, cens), rl.fit_weibull(tau, cens), rl.fit_hawkes(t_tr, T_split)
    brng = np.random.default_rng(seed + 1)
    kb = [rl.fit_weibull(tau[brng.integers(0, len(tau), len(tau))])["k"] for _ in range(1000)]
    fol = tau[tau < gap_min]
    aic = {"exponential": rl.aic(m_exp), "weibull": rl.aic(m_wb), "hawkes": rl.aic(m_hk)}
    clustered = aic["hawkes"] < aic["exponential"]
    return {"train_events": int(len(t_tr)), "aic": aic, "weibull_k": m_wb["k"],
            "weibull_k_ci": [float(np.quantile(kb, 0.025)), float(np.quantile(kb, 0.975))],
            "hawkes_branching": m_hk["branching"],
            "watch_window_min": float(np.quantile(fol, 0.9)) if len(fol) else None,
            "verdict": "clustered: split first vs follow-on" if clustered else "not clustered: treat events alike"}


def model_check(df, sig, src, onsets, t_on, tmin, ep_first, mask_k, horizon_min, gap_min, perms):
    """Sensor-only GBM on first-in-episode events, forward-chaining folds as in
    pipeline/hybrid_run.py, against a per-block rotation chance null."""
    n = len(df)
    h = horizon_min // 2
    F = hr.features(df, sig, src)
    Xcat = pd.get_dummies(df[ld.CATEGORICAL].astype(str), prefix="grade").astype(float)
    Xcat[ld.BINARY] = df[ld.BINARY].astype(float)
    X = pd.concat([F, Xcat], axis=1)
    label, elig = lb.make_labels(n, onsets, h, mask_k)
    bounds = hy.episode_safe_bounds(n, onsets, tmin, hr.K_FOLDS, gap_min)
    rows = np.arange(n)
    score, fold_of = np.full(n, np.nan), np.full(n, -1)
    for f in range(1, hr.K_FOLDS):
        b0, b1 = bounds[f], bounds[f + 1]
        mtr, mte = (rows < b0 - h) & elig, (rows >= b0) & (rows < b1) & elig
        fold_of[(rows >= b0) & (rows < b1)] = f
        score[mte] = hy.fit_plain_gbm(X[mtr], label[mtr], hr.SEED).predict_proba(X[mte])[:, 1]
    test = (fold_of > 0) & elig
    ti = np.where(test)[0]
    y = label[ti]
    nxt = onsets[np.minimum(np.searchsorted(onsets, ti, side="right"), len(onsets) - 1)]
    first_set = set(onsets[ep_first].tolist())
    nf = np.array([o in first_set for o in nxt])
    blk = [np.where((fold_of == f) & elig)[0] for f in range(1, hr.K_FOLDS)]
    yfull = np.zeros(n, int)
    yfull[ti] = y
    s_full = np.where(test, score, 0.0)
    nrng = np.random.default_rng(hr.SEED + 2000)
    out = {}
    for name, min_lead in (("lead_ge_0", 0), ("lead_ge_10min", 5)):
        keep = (y == 0) | (nf & ((nxt - ti) >= min_lead))
        msk = np.zeros(n, bool)
        msk[ti] = keep
        obs = float(hy.average_precision_score(y[keep], score[ti][keep]))
        nl = hy.shift_null_ap(yfull, s_full, blk, msk, perms, nrng)
        out[name] = {"pr_auc": obs, "base_rate": float(y[keep].mean()), "null_mean": float(nl.mean()),
                     "p": float((1 + (nl >= obs).sum()) / (1 + perms))}
    ok = all(v["p"] < 0.05 and v["pr_auc"] > v["base_rate"] for v in out.values())
    out["verdict"] = "first-break signal above chance: screening only" if ok else "no first-break signal"
    return out


def triage(src, grade="x28", gap_min=120, horizon_min=20, perms=5000, model=False):
    ld.CATEGORICAL = grade
    run.B = perms
    run.EPISODE_GAP_MIN = gap_min
    rng = np.random.default_rng(run.SEED)
    df = load_csv(src)
    sig = ld.signal_columns(df)
    n = len(df)
    onsets = ld.break_onsets(df["y"])
    tmin = (df["time"] - df["time"].iloc[0]).dt.total_seconds().values / 60
    t_on = tmin[onsets]
    ep_id, ep_first = rl.episodes(t_on, gap_min)
    split = int(n * (1 - run.TEST_FRAC))
    tr = np.arange(n) < split
    out = {"source": os.path.basename(src), "rows": n, "signals": len(sig),
           "cost": cost_card(df, tmin, onsets),
           "clustering": clustering_card(t_on, tmin[split], gap_min, run.SEED)}

    # routing: one row per event
    out["episodes"] = {"episodes": int(ep_first.sum()), "first_in_episode": int(ep_first.sum()),
                       "follow_on": int((~ep_first).sum()), "gap_min": gap_min}
    out["routing"] = [{"event": i + 1, "time": df["time"].iloc[o].strftime("%Y-%m-%d %H:%M"),
                       "episode": int(ep_id[i]) + 1,
                       "kind": "first" if ep_first[i] else "follow-on",
                       "route": "sensor screening" if ep_first[i] else "hazard / watch window"}
                      for i, o in enumerate(onsets)]

    # detector screen: break-level and episode-level tests from pipeline/run.py
    Z = run.zscore(df, sig, df.index[tr])
    mask_k, _ = lb.restart_mask_len(Z, onsets[onsets < split])
    F = hr.features(df, sig, src)
    W = horizon_min // 2
    label_w, elig_w = lb.make_labels(n, onsets, W, mask_k)
    bounds = np.linspace(0, n, run.N_BLOCKS + 1).astype(int)
    first_set = set(onsets[ep_first].tolist())
    sd = run.score_detectors(df, F, sig, onsets, label_w, elig_w, W, bounds,
                             list(range(1, run.N_BLOCKS)), rng, first_set)
    dets = [{"id": d["id"], "lift": d["lift"], "q": d["q"], "q_ep": d["q_ep"],
             "covered_ep": d["covered_ep"], "survives": d["survives"], "survives_ep": d["survives_ep"],
             "label": rule_label(d), "action": ACTION[rule_label(d)]} for d in sd["dets"]]
    out["mask_rows"] = int(mask_k)
    out["detectors"] = {"rules": len(dets), "eval_events": int(len(sd["ev"]["onsets"])),
                        "eval_episodes": int(len(sd["ev"]["ep_on"])),
                        "survive_per_event": sd["n_surv"], "survive_per_episode": sd["n_surv_ep"],
                        "list": dets}

    # SHAP vs lift: the importance model of pipeline/run.py at the same horizon
    label, elig = lb.make_labels(n, onsets, W, mask_k)
    Xcat = pd.get_dummies(df[ld.CATEGORICAL].astype(str), prefix="grade").astype(float)
    Xcat[ld.BINARY] = df[ld.BINARY].astype(float)
    gbm, X, mte, *_ = run.fit_models(F, Xcat, label, elig, tr, ~tr)
    contrib = gbm.booster_.predict(X[mte], pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=X.columns)
    shap = imp.groupby([c.split("__")[0] for c in imp.index]).sum().reindex(sig).fillna(0.0)
    by_sig = {s: [d for d in sd["dets"] if d["signal"] == s and d["lift"] is not None] for s in sig}
    survives = {s: any(d["survives"] for d in ds) for s, ds in by_sig.items()}
    pairs = [(shap[s], max(d["lift"] for d in ds)) for s, ds in by_sig.items() if ds]
    top10 = list(shap.rank(ascending=False, method="first").sort_values().index[:10])
    out["shap_vs_lift"] = {"top10": top10, "top10_failing": [s for s in top10 if not survives[s]],
                           "spearman": float(spearmanr(*zip(*pairs)).correlation)}

    if model:
        out["first_break_model"] = model_check(df, sig, src, onsets, t_on, tmin, ep_first, mask_k,
                                               horizon_min, gap_min, 1000)
    return out


def card(o):
    c, k, e, d, s = o["cost"], o["clustering"], o["episodes"], o["detectors"], o["shap_vs_lift"]
    L = [f"TRIAGE  {o['source']}  ({o['rows']:,} rows, {o['signals']} sensors)", "",
         "1. COST (inferred from timestamp gaps)",
         f"   {c['events']} events over {c['days']:.1f} days; {c['events_with_gap']} followed by a gap of {c['sampling_gaps']}",
         f"   {c['downtime_min_per_event']:.1f} min per event, {c['downtime_hours']:.1f} h total, "
         f"{c['line_hours_per_day']:.1f} line-hours a day", "",
         "2. CLUSTERING (fit on the first 75% of the record)",
         "   AIC  exponential {exponential:.1f}  weibull {weibull:.1f}  hawkes {hawkes:.1f}".format(**k["aic"]),
         f"   Weibull k {k['weibull_k']:.2f} (CI {k['weibull_k_ci'][0]:.2f} to {k['weibull_k_ci'][1]:.2f}), "
         f"Hawkes branching {k['hawkes_branching']:.2f}",
         f"   watch window (90th pct of follow-on spacing): {k['watch_window_min']:.0f} min",
         f"   -> {k['verdict']}", "",
         f"3. ROUTING  {c['events']} events -> {e['episodes']} episodes (gap {e['gap_min']} min)",
         f"   {e['first_in_episode']} first-in-episode -> sensor screening",
         f"   {e['follow_on']} follow-on        -> hazard / watch window",
         "   (per-event table in the JSON output)", "",
         f"4. DETECTORS  {d['rules']} rules, {d['eval_events']} scored events, {d['eval_episodes']} episodes",
         f"   survive per event: {d['survive_per_event']}   survive per episode: {d['survive_per_episode']}"]
    flagged = [x for x in d["list"] if x["label"] != NOISE]
    for x in flagged:
        L.append(f"   {x['id']:<14} lift {x['lift']:.1f}  q {x['q']:.3f}  q_ep {x['q_ep']:.2f}  "
                 f"{x['label']}: {x['action']}")
    L.append(f"   other {d['rules'] - len(flagged)} rules: {NOISE}")
    L += ["", "5. SHAP vs LIFT",
          f"   {len(s['top10_failing'])} of the top-10 SHAP signals fail the test "
          f"(model leans on it; does not precede events): {', '.join(s['top10_failing'])}",
          f"   Spearman SHAP vs best lift: {s['spearman']:.2f}"]
    if "first_break_model" in o:
        m = o["first_break_model"]
        L += ["", "6. FIRST-BREAK MODEL (sensor GBM, rotation chance null)"]
        for nm, lab in (("lead_ge_0", "lead >= 0     "), ("lead_ge_10min", "lead >= 10 min")):
            v = m[nm]
            L.append(f"   {lab} PR-AUC {v['pr_auc']:.3f} vs base rate {v['base_rate']:.3f} "
                     f"({v['pr_auc'] / v['base_rate']:.1f}x), p = {v['p']:.3f}")
        L.append(f"   -> {m['verdict']}")
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("csv")
    p.add_argument("--grade", default="x28")
    p.add_argument("--gap-min", type=float, default=120)
    p.add_argument("--horizon-min", type=int, default=20)
    p.add_argument("--perms", type=int, default=5000)
    p.add_argument("--model", action="store_true", help="add the first-break sensor-model check (slower)")
    p.add_argument("--out", default="triage.json")
    a = p.parse_args(argv)
    o = triage(a.csv, a.grade, a.gap_min, a.horizon_min, a.perms, a.model)
    print(card(o))
    with open(a.out, "w") as f:
        json.dump(o, f, indent=1)
    print(f"\nwrote {a.out}")
    return o


if __name__ == "__main__":
    main()
