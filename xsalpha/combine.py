"""Combines the deflated-Sharpe survivors with the two alternative-data
signals into one portfolio, attributes its daily return to sector, size
and momentum exposure, and prices the unexplained residual under the
existing square-root market-impact cost model.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from . import factors as xsfactors
from . import ic as xsic
from . import impact as xsimpact
from . import portfolio as xsportfolio
from .neutralize import sector_neutralize

# Half the bid-ask spread, charged on every dollar traded (not held), in
# addition to the square-root impact cost already in impact.py. 5bps is a
# representative round-trip spread for a liquid large-cap name; half of
# it (2.5bps one-way) is the standard crossing-cost convention.
HALF_SPREAD_BPS = 0.00025


def combined_weights(
    panel: dict,
    survivor_scores: Dict[str, pd.DataFrame],
    altdata_scores: Dict[str, pd.DataFrame],
    test_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Sector-neutralizes every signal (survivors and alt-data alike),
    then equal-weight z-score combines all of them, the same
    `composite_score`/`dollar_neutral_weights` path the 3-survivor
    portfolio already uses."""
    all_scores = {**survivor_scores, **altdata_scores}
    neutral = {name: sector_neutralize(df, panel["sector"]) for name, df in all_scores.items()}
    combo = xsportfolio.composite_score(neutral).loc[test_dates]
    return xsportfolio.dollar_neutral_weights(combo)


def combined_gross_return(weights: pd.DataFrame, fwd_ret_1d: pd.DataFrame) -> pd.Series:
    """Daily return of the unit-gross combined portfolio, the same
    weights-times-forward-return convention impact.simulate_at_size uses."""
    common_cols = weights.columns.intersection(fwd_ret_1d.columns)
    w = weights[common_cols]
    fwd = fwd_ret_1d.loc[w.index, common_cols]
    return (w * fwd).sum(axis=1)


def net_return_at_size(
    panel: dict, weights: pd.DataFrame, fwd_ret_1d: pd.DataFrame, gross_dollars: float
) -> pd.Series:
    """Daily net-of-cost return of the combined portfolio at a given
    gross dollar size, reusing impact.py's cost model exactly."""
    adv = xsimpact.dollar_adv(panel)
    vol = xsimpact.realized_vol(panel)
    dollars = weights * gross_dollars
    trade = dollars.diff().fillna(dollars.iloc[0])
    common_cols = dollars.columns.intersection(fwd_ret_1d.columns).intersection(adv.columns)
    dollars = dollars[common_cols]
    trade = trade[common_cols]
    fwd = fwd_ret_1d.loc[dollars.index, common_cols]
    adv_c = adv.loc[dollars.index, common_cols]
    vol_c = vol.loc[dollars.index, common_cols]

    gross_pnl = (dollars * fwd).sum(axis=1)
    impact_frac = xsimpact.impact_cost_fraction(trade, adv_c, vol_c)
    impact_dollars = (impact_frac * trade.abs()).sum(axis=1)
    spread_dollars = (HALF_SPREAD_BPS * trade.abs()).sum(axis=1)
    net_pnl = gross_pnl - impact_dollars - spread_dollars
    return net_pnl / gross_dollars


def run_attribution(
    panel: dict,
    survivor_scores: Dict[str, pd.DataFrame],
    altdata_scores: Dict[str, pd.DataFrame],
    test_dates: pd.DatetimeIndex,
    gross_dollars_for_cost: float,
) -> dict:
    weights = combined_weights(panel, survivor_scores, altdata_scores, test_dates)
    fwd_ret_1d = xsic.forward_return(panel["close"], horizon=1)

    gross_ret = combined_gross_return(weights, fwd_ret_1d)
    factors_df = xsfactors.build_factor_matrix(panel, fwd_ret_1d, gross_ret.index)

    fitted, resid_gross, r2, beta = xsfactors.attribute(gross_ret, factors_df)

    net_ret = net_return_at_size(panel, weights, fwd_ret_1d, gross_dollars_for_cost)
    net_ret_common = net_ret.reindex(fitted.index)
    resid_net = net_ret_common - fitted

    return {
        "weights": weights,
        "gross_ret": gross_ret,
        "factors": factors_df,
        "fitted": fitted,
        "resid_gross": resid_gross,
        "resid_net": resid_net,
        "r2_explained": r2,
        "beta": beta,
        "gross_sharpe_residual": xsfactors.annualized_sharpe(resid_gross),
        "net_sharpe_residual": xsfactors.annualized_sharpe(resid_net),
        "gross_sharpe_combined": xsfactors.annualized_sharpe(gross_ret),
    }
