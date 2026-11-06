"""Load the paper-machine sheet-break data and flag break episodes."""
import numpy as np
import pandas as pd

CATEGORICAL = "x28"   # grade / product code (8 levels)
BINARY = "x61"


def load(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], format="%m/%d/%y %H:%M")
    df = df.sort_values("time").reset_index(drop=True)
    return df


def signal_columns(df):
    return [c for c in df.columns if c.startswith("x") and c not in (CATEGORICAL, BINARY)]


def break_onsets(y):
    """Row index of each break onset; consecutive y=1 rows count as one event."""
    y = np.asarray(y).astype(int)
    prev = np.r_[0, y[:-1]]
    return np.where((y == 1) & (prev == 0))[0]


def sampling_gaps(df, nominal_min=2.0):
    d = df["time"].diff().dt.total_seconds().div(60).fillna(nominal_min)
    return np.where(d.values > nominal_min)[0]
