"""Sector neutralization: demean each day's cross-section within each
sector, so a signal cannot be a disguised sector bet."""
from __future__ import annotations

import pandas as pd


def sector_neutralize(score: pd.DataFrame, sector: pd.Series) -> pd.DataFrame:
    """score: (dates x tickers). sector: ticker -> sector name.
    Returns a same-shaped DataFrame where, for every date and every
    sector, the mean of the neutralized scores over that sector's
    tickers is (numerically) zero.
    """
    out = score.copy()
    for sec_name, tickers in sector.groupby(sector).groups.items():
        cols = [t for t in tickers if t in out.columns]
        if not cols:
            continue
        block = out[cols]
        out[cols] = block.sub(block.mean(axis=1), axis=0)
    return out


def check_sector_neutral(neutral_score: pd.DataFrame, sector: pd.Series, tol: float = 1e-9) -> float:
    """Returns the max absolute per-sector, per-day mean (should be ~0)."""
    max_abs = 0.0
    for sec_name, tickers in sector.groupby(sector).groups.items():
        cols = [t for t in tickers if t in neutral_score.columns]
        if not cols:
            continue
        means = neutral_score[cols].mean(axis=1).dropna()
        if len(means):
            max_abs = max(max_abs, means.abs().max())
    return float(max_abs)
