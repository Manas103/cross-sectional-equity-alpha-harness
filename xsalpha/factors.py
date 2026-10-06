"""Sector, size and momentum factor returns, and an OLS attribution of a
portfolio's daily return onto those exposures.

Sector factors are the simplest honest construction available from the
data this repository actually returns from `simulate_panel`: the
equal-weight average daily return of the stocks in each sector, the same
way a real sector index return is built. `simulate_panel` does not expose
its own internal `sector_ret` array (a local variable used only to build
the price panel), so this is an empirical reconstruction from realized
per-stock returns, not a readout of the generator's hidden parameters.

Size and momentum factors are canonical decile long-short portfolios
(bottom decile minus top decile by trailing dollar ADV for size, top
decile minus bottom decile by trailing 60-day return for momentum),
equal-weighted within each decile, built causally (only data strictly
before each date).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import impact as xsimpact


def sector_factor_returns(panel: dict) -> pd.DataFrame:
    """One column per sector: that sector's equal-weight daily return."""
    returns = panel["returns"]
    sector = panel["sector"]
    cols = {}
    for sec_name, tickers in sector.groupby(sector).groups.items():
        present = [t for t in tickers if t in returns.columns]
        if present:
            cols[sec_name] = returns[present].mean(axis=1)
    return pd.DataFrame(cols)


def _decile_long_short(score: pd.DataFrame, fwd_ret: pd.DataFrame, long_top: bool) -> pd.Series:
    """Equal-weight top-decile-minus-bottom-decile (or reverse) daily
    return from a causal cross-sectional score, using the SAME day's
    1-day forward return (close[t+1]/close[t]-1), matching the
    convention impact.simulate_at_size already uses elsewhere in this
    repository."""
    out = {}
    for dt in score.index:
        s = score.loc[dt].dropna()
        if len(s) < 20:
            continue
        r = fwd_ret.loc[dt].reindex(s.index)
        decile = (s.rank(pct=True) * 10).clip(upper=9.999).astype(int)
        top = r[decile == 9].mean()
        bottom = r[decile == 0].mean()
        if pd.isna(top) or pd.isna(bottom):
            continue
        out[dt] = (top - bottom) if long_top else (bottom - top)
    return pd.Series(out).sort_index()


def size_factor_return(panel: dict, fwd_ret_1d: pd.DataFrame) -> pd.Series:
    """Small-minus-big: long the smallest decile by trailing dollar ADV,
    short the largest, the standard SMB sign convention."""
    adv = xsimpact.dollar_adv(panel)
    return _decile_long_short(adv, fwd_ret_1d, long_top=False)


def momentum_factor_return(panel: dict, fwd_ret_1d: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """Up-minus-down: long the top decile by trailing `lookback`-day
    return, short the bottom decile, the standard UMD sign convention."""
    close = panel["close"]
    trailing = (close / close.shift(lookback) - 1.0).shift(1)  # causal: known before today
    return _decile_long_short(trailing, fwd_ret_1d, long_top=True)


def build_factor_matrix(
    panel: dict, fwd_ret_1d: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    sector_ret = sector_factor_returns(panel).reindex(dates)
    size_ret = size_factor_return(panel, fwd_ret_1d).reindex(dates).rename("size")
    mom_ret = momentum_factor_return(panel, fwd_ret_1d).reindex(dates).rename("momentum")
    return pd.concat([sector_ret, size_ret, mom_ret], axis=1).dropna()


def attribute(
    portfolio_ret: pd.Series, factors: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, float, np.ndarray]:
    """OLS of portfolio_ret on factors, with an intercept fit only to
    get unbiased factor loadings. The intercept is deliberately *not*
    subtracted out of the residual: it is the portfolio's own average
    return net of factor tilts (its alpha), not something sector, size
    or momentum exposure explains, so folding it into "explained return"
    would overstate how much of the return the factors actually account
    for. `fitted` below is the factor-loadings-only component; `residual`
    is everything else (intercept plus the OLS residual), which is what
    feeds the residual Sharpe. Returns (fitted, residual, pct_of_return_
    explained, beta); pct_of_return_explained is sum(fitted)/sum(actual),
    a return-based attribution, not a variance-based R^2 (an already
    sector-neutral, dollar-neutral portfolio's day-to-day variance is
    mostly idiosyncratic by construction, so a variance R^2 against
    factor returns would be a different, much smaller number answering a
    different question).
    """
    common = portfolio_ret.index.intersection(factors.index)
    y = portfolio_ret.loc[common].to_numpy()
    X_factors = factors.loc[common].to_numpy()
    X = np.column_stack([X_factors, np.ones(len(common))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    beta_factors = beta[:-1]

    fitted_vals = X_factors @ beta_factors
    resid_vals = y - fitted_vals

    total_return = float(np.sum(y))
    explained_return = float(np.sum(fitted_vals))
    pct_explained = explained_return / total_return if total_return != 0 else float("nan")

    fitted = pd.Series(fitted_vals, index=common)
    resid = pd.Series(resid_vals, index=common)
    return fitted, resid, pct_explained, beta


def annualized_sharpe(x: pd.Series, periods_per_year: float = 252.0) -> float:
    x = x.dropna()
    if len(x) < 2 or x.std() == 0:
        return 0.0
    return float(x.mean() / x.std() * np.sqrt(periods_per_year))
