"""Lift, precision, permutation p-values and Benjamini-Hochberg."""
import numpy as np


def lift_from_counts(fire_pre, n_pre, fire_nom, n_nom):
    """Lift = P(fire | pre-break window) / P(fire | nominal window)."""
    p_pre = fire_pre / n_pre
    p_nom = fire_nom / n_nom
    return p_pre / p_nom if p_nom > 0 else np.inf


def precision_from_counts(fire_pre, fire_nom):
    """Share of fired windows that were pre-break windows."""
    tot = fire_pre + fire_nom
    return fire_pre / tot if tot else 0.0


def perm_pvalue(observed, null):
    """p = (1 + #{null >= observed}) / (1 + B)."""
    null = np.asarray(null)
    return (1 + np.sum(null >= observed - 1e-12)) / (1 + len(null))


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (step-up, monotone)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    return out


def window_any(fire, w):
    """any_fire[t] = 1 if fire is true anywhere in rows (t-w, t] (past-only)."""
    f = np.asarray(fire).astype(int)
    c = np.r_[0, np.cumsum(f)]
    idx = np.arange(len(f))
    lo = np.maximum(0, idx + 1 - w)
    return (c[idx + 1] - c[lo]) > 0
