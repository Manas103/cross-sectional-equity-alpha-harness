"""Daily rank information coefficient, computed out of sample under the
purged walk-forward folds in splits.py.

For each fold, a single scalar (the sign of the signal's mean training-
period IC) is estimated from the training window only and applied,
unchanged, to every test day in that fold. This is the "retrained"
step: a signal that was rank-correlated with forward returns one way in
an early regime and the opposite way in a later one will show that in
`fold_signs` (see scripts/run_signal_mining.py), and it is estimated
strictly from data at or before each fold's train_idx.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .splits import Fold, HORIZON


def forward_return(close: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    return close.shift(-horizon) / close - 1.0


def daily_rank_ic(score: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """Spearman rank correlation between score and forward return, one
    value per date, computed independently per row (no cross-date
    information used)."""
    out = {}
    for dt in score.index:
        s = score.loc[dt]
        r = fwd_ret.loc[dt]
        mask = s.notna() & r.notna()
        if mask.sum() < 10:
            continue
        sr = s[mask].rank()
        rr = r[mask].rank()
        sr_c = sr - sr.mean()
        rr_c = rr - rr.mean()
        denom = np.sqrt((sr_c ** 2).sum() * (rr_c ** 2).sum())
        out[dt] = float((sr_c * rr_c).sum() / denom) if denom > 0 else np.nan
    return pd.Series(out).dropna()


def oos_ic_series(raw_ic: pd.Series, folds: List[Fold], dates: pd.DatetimeIndex,
                   thin: int = 1) -> Tuple[pd.Series, List[int]]:
    """raw_ic: full-sample daily IC (index = date), computed once from the
    un-flipped score. Returns (oos_ic, fold_signs) where oos_ic is the
    concatenation of each fold's sign-adjusted test-day IC, and
    fold_signs[k] is the training-estimated sign applied in fold k.

    `thin`: keep only every `thin`-th test day *within each fold*
    (subsampled independently per fold, so the seam between folds never
    creates an artificial overlap). Since the IC label is a HORIZON-day
    forward return, thin=HORIZON gives a set of non-overlapping-label
    observations, which is what the Sharpe/skew/kurtosis inputs to the
    deflated Sharpe ratio assume; thin=1 keeps every (overlapping) day.
    """
    pieces = []
    signs = []
    for fold in folds:
        train_dates = dates[list(fold.train_idx)]
        test_dates = dates[list(fold.test_idx)]
        train_ic = raw_ic.reindex(train_dates).dropna()
        sign = 1.0 if len(train_ic) == 0 or train_ic.mean() >= 0 else -1.0
        signs.append(sign)
        test_ic = raw_ic.reindex(test_dates).dropna() * sign
        if thin > 1:
            test_ic = test_ic.iloc[::thin]
        pieces.append(test_ic)
    if pieces:
        return pd.concat(pieces).sort_index(), signs
    return pd.Series(dtype=float), signs
