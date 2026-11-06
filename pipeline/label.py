"""Forward-looking labels with break-row and restart masking."""
import numpy as np


def restart_mask_len(X, onsets, pre=15, max_k=30, n_sd=2.0, calm=3, q=0.9):
    """Rows after a break until the restart transient has died out.

    X: standardised signal matrix (rows x signals). The transient score at row
    t is the mean |z_t - z_{t-1}| across signals (how fast the line is moving).
    For each break the calm limit is mean + n_sd * sd of that score over the
    `pre` rows before onset. Recovery = first post-break row that starts a run
    of `calm` consecutive rows under the limit. Returns the q-quantile of the
    recovery lengths (rows) and the per-break lengths.
    """
    dz = np.r_[np.zeros((1, X.shape[1])), np.abs(np.diff(X, axis=0))].mean(1)
    lens = []
    n = len(X)
    for o in onsets:
        if o - pre < 0 or o + 1 + max_k >= n:
            continue
        p = dz[o - pre:o]
        lim = p.mean() + n_sd * p.std()
        post = dz[o + 1:o + 1 + max_k]
        k = max_k
        for i in range(len(post) - calm + 1):
            if (post[i:i + calm] <= lim).all():
                k = i + 1
                break
        lens.append(k)
    return (int(np.ceil(np.quantile(lens, q))) if lens else max_k), lens


def make_labels(n, onsets, h_rows, mask_k, extra_shift=0):
    """label[t]=1 if a break onset lies in (t, t+h_rows] (shifted by extra_shift).

    Returns (label, eligible). Break rows and the mask_k rows after each break
    are ineligible. extra_shift>0 moves the window one step further from the
    break (leakage check: skill should not jump).
    """
    label = np.zeros(n, dtype=int)
    eligible = np.ones(n, dtype=bool)
    for o in onsets:
        lo = max(0, o - h_rows - extra_shift)
        hi = o - extra_shift  # exclusive
        if hi > lo:
            label[lo:hi] = 1
        eligible[o:min(n, o + 1 + mask_k)] = False
    label[~eligible] = 0
    return label, eligible
