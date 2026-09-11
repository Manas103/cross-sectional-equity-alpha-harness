"""WorldQuant-101-style cross-sectional and time-series operators.

Every function takes and returns a (dates x tickers) DataFrame. Cross-
sectional operators (`rank`, `scale`, `demean`) act along the ticker axis
for each date independently; time-series operators (`ts_*`, `delay`,
`decay_linear`) act along the date axis for each ticker independently and
never look forward (a rolling window ending at row t uses only rows
<= t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank in [0, 1], per date."""
    return df.rank(axis=1, pct=True)


def scale(df: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    """Rescale each date's cross-section so the sum of absolute values is a."""
    denom = df.abs().sum(axis=1)
    denom = denom.replace(0.0, np.nan)
    return df.div(denom, axis=0) * a


def demean(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0)


def delay(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.shift(n)


def delta(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df - df.shift(n)


def ts_sum(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).sum()


def ts_mean(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).mean()


def ts_std(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).std()


def ts_min(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).min()


def ts_max(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).max()


def ts_argmax(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).apply(lambda x: float(np.argmax(x)), raw=True)


def ts_argmin(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).apply(lambda x: float(np.argmin(x)), raw=True)


def ts_rank(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).apply(
        lambda x: (pd.Series(x).rank(pct=True).iloc[-1]), raw=False
    )


def decay_linear(df: pd.DataFrame, n: int) -> pd.DataFrame:
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()

    def _wavg(x):
        return float(np.dot(x, weights))

    return df.rolling(n, min_periods=n).apply(_wavg, raw=True)


def correlation(a: pd.DataFrame, b: pd.DataFrame, n: int) -> pd.DataFrame:
    return a.rolling(n, min_periods=n).corr(b)


def covariance(a: pd.DataFrame, b: pd.DataFrame, n: int) -> pd.DataFrame:
    return a.rolling(n, min_periods=n).cov(b)


def sign(df: pd.DataFrame) -> pd.DataFrame:
    return np.sign(df)


def signed_power(df: pd.DataFrame, p: float) -> pd.DataFrame:
    return np.sign(df) * (df.abs() ** p)


def clip_lower(df: pd.DataFrame, lower: float = 0.0) -> pd.DataFrame:
    return df.clip(lower=lower)
