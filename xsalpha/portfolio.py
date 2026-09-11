"""Cross-sectional long-short portfolio construction from the surviving
signals' sector-neutral scores."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .neutralize import sector_neutralize


def composite_score(neutral_scores: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight average of z-scored (per day, cross-sectionally)
    sector-neutral scores across the surviving signals."""
    zscored = []
    for name, df in neutral_scores.items():
        mu = df.mean(axis=1)
        sd = df.std(axis=1).replace(0.0, np.nan)
        zscored.append(df.sub(mu, axis=0).div(sd, axis=0))
    stacked = sum(zscored) / len(zscored)
    return stacked


def dollar_neutral_weights(score: pd.DataFrame) -> pd.DataFrame:
    """Rank-based long-short weights, dollar-neutral (sums to ~0) and
    unit gross exposure (sum of |weight| == 1) each day."""
    ranked = score.rank(axis=1, pct=True) - 0.5  # in [-0.5, 0.5], mean 0 by construction
    gross = ranked.abs().sum(axis=1).replace(0.0, np.nan)
    return ranked.div(gross, axis=0)


def build_survivor_portfolio(panel: dict, survivor_scores: Dict[str, pd.DataFrame],
                              test_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Returns a weights DataFrame (test_dates x tickers), sector-
    neutralized and dollar-neutral, unit gross exposure per day."""
    neutral = {name: sector_neutralize(df, panel["sector"]) for name, df in survivor_scores.items()}
    combo = composite_score(neutral).loc[test_dates]
    return dollar_neutral_weights(combo)
