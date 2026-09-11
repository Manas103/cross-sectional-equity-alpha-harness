"""Square-root market-impact cost model and the capacity sweep.

Cost is charged on every day's traded notional (the change in dollar
position from the prior day), in the standard square-root-of-
participation form used throughout the market-impact literature
(Almgren-Chriss and Kyle-lambda-style models both converge on this
shape): cost as a fraction of traded notional scales with
sqrt(trade size / average daily dollar volume), so cost in dollars
scales as trade_size^1.5, not trade_size^1 or trade_size^2.

IMPACT_COEF is a realistic order-of-magnitude constant (in the same
spirit as futures-strategy-backtester's illustrative 0.08 coefficient),
chosen once, before any capacity sweep was run, not tuned afterward to
land on a particular crossing point.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from . import db as xsdb

IMPACT_COEF = 0.6  # dimensionless, multiplies daily_vol * sqrt(participation)


def dollar_adv(panel: dict, window: int = 60, db_path: str | None = None) -> pd.DataFrame:
    """Trailing average daily dollar volume, causal (uses only data
    strictly before each date). When db_path is given, this is computed
    by the real DuckDB window-function query in xsalpha/db.py; otherwise
    falls back to an equivalent pandas rolling computation."""
    if db_path is not None:
        return xsdb.sql_dollar_adv(db_path, window)
    dollar_vol = panel["volume"] * panel["close"]
    return dollar_vol.rolling(window, min_periods=window).mean().shift(1)


def realized_vol(panel: dict, window: int = 60) -> pd.DataFrame:
    """Trailing realized daily return volatility, causal."""
    return panel["returns"].rolling(window, min_periods=window).std().shift(1)


def impact_cost_fraction(trade_dollars: pd.DataFrame, adv_dollars: pd.DataFrame,
                          vol: pd.DataFrame, coef: float = IMPACT_COEF) -> pd.DataFrame:
    """Returns cost as a fraction of the *portfolio's* gross dollar size
    G for each date (sum over tickers of dollar cost / G is done by the
    caller after multiplying by trade_dollars in absolute terms)."""
    participation = (trade_dollars.abs() / adv_dollars).clip(lower=0.0)
    return coef * vol * np.sqrt(participation)


def simulate_at_size(panel: dict, weights: pd.DataFrame, fwd_ret_1d: pd.DataFrame,
                      gross_dollars: float, adv: pd.DataFrame, vol: pd.DataFrame) -> Dict[str, float]:
    """weights: (dates x tickers), dollar-neutral, unit gross exposure.
    Returns gross/net Sharpe and mean daily cost (bps of gross size) at
    this position size."""
    dollars = weights * gross_dollars
    trade = dollars.diff().fillna(dollars.iloc[0])
    common_cols = dollars.columns.intersection(fwd_ret_1d.columns).intersection(adv.columns)
    dollars = dollars[common_cols]
    trade = trade[common_cols]
    fwd = fwd_ret_1d.loc[dollars.index, common_cols]
    adv_c = adv.loc[dollars.index, common_cols]
    vol_c = vol.loc[dollars.index, common_cols]

    gross_pnl = (dollars * fwd).sum(axis=1)
    cost_frac = impact_cost_fraction(trade, adv_c, vol_c)
    cost_dollars = (cost_frac * trade.abs()).sum(axis=1)
    net_pnl = gross_pnl - cost_dollars

    gross_ret = gross_pnl / gross_dollars
    net_ret = net_pnl / gross_dollars

    def sharpe(x: pd.Series) -> float:
        x = x.dropna()
        if x.std() == 0 or len(x) < 2:
            return 0.0
        return float(x.mean() / x.std() * np.sqrt(252))

    return {
        "gross_dollars": gross_dollars,
        "gross_sharpe": sharpe(gross_ret),
        "net_sharpe": sharpe(net_ret),
        "mean_daily_cost_bps_of_gross": float((cost_dollars / gross_dollars).mean() * 1e4),
    }


def capacity_sweep(panel: dict, weights: pd.DataFrame, fwd_ret_1d: pd.DataFrame,
                    sizes: List[float], db_path: str | None = None) -> pd.DataFrame:
    adv = dollar_adv(panel, db_path=db_path)
    vol = realized_vol(panel)
    rows = [simulate_at_size(panel, weights, fwd_ret_1d, g, adv, vol) for g in sizes]
    return pd.DataFrame(rows)


def find_capacity_crossing(sweep: pd.DataFrame, threshold: float = 0.5) -> float:
    """Linear-interpolate the gross dollar size at which net_sharpe first
    crosses below `threshold` as size increases. Returns np.inf if the
    threshold is never crossed within the sweep."""
    sweep = sweep.sort_values("gross_dollars").reset_index(drop=True)
    for i in range(1, len(sweep)):
        s0, s1 = sweep.loc[i - 1, "net_sharpe"], sweep.loc[i, "net_sharpe"]
        g0, g1 = sweep.loc[i - 1, "gross_dollars"], sweep.loc[i, "gross_dollars"]
        if s0 >= threshold > s1:
            frac = (s0 - threshold) / (s0 - s1)
            return float(g0 + frac * (g1 - g0))
    return float("inf")
