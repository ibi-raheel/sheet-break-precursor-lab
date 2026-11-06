"""Past-only rolling features. Every value at row t uses rows <= t only."""
import numpy as np
import pandas as pd

WINDOWS = (5, 15)  # rows: 10 and 30 minutes at 2-minute sampling


def build(df, signals, windows=WINDOWS):
    out = {}
    for s in signals:
        v = df[s].astype(float)
        out[f"{s}"] = v
        for w in windows:
            r = v.rolling(w, min_periods=w)
            out[f"{s}__mean{w}"] = r.mean()
            out[f"{s}__std{w}"] = r.std()
            out[f"{s}__dev{w}"] = v - r.median()
            # least-squares slope over the window = cov(x,v)/var(x), computed in closed form
            xs = np.arange(w) - (w - 1) / 2
            out[f"{s}__slope{w}"] = v.rolling(w, min_periods=w).apply(
                lambda a, xs=xs: np.dot(xs, a) / np.dot(xs, xs), raw=True)
    F = pd.DataFrame(out)
    return F


def base_signal(feature_name):
    return feature_name.split("__")[0]
