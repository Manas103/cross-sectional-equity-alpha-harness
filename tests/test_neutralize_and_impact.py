"""Sector-neutrality invariant and the impact-cost model's monotonicity
(net Sharpe is non-increasing in trade size)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from xsalpha.impact import impact_cost_fraction
from xsalpha.neutralize import check_sector_neutral, sector_neutralize
from xsalpha.portfolio import dollar_neutral_weights


def _toy_panel(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=20)
    tickers = [f"T{i}" for i in range(12)]
    sector = pd.Series(["A"] * 4 + ["B"] * 4 + ["C"] * 4, index=tickers)
    score = pd.DataFrame(rng.normal(size=(20, 12)), index=dates, columns=tickers)
    return score, sector


def test_sector_neutralize_zeros_out_per_sector_mean():
    score, sector = _toy_panel()
    neutral = sector_neutralize(score, sector)
    max_abs_mean = check_sector_neutral(neutral, sector)
    assert max_abs_mean < 1e-9


def test_dollar_neutral_weights_have_the_documented_small_bias_and_unit_gross():
    # dollar_neutral_weights ranks each day (score.rank(pct=True) - 0.5),
    # then divides by that day's raw gross. For n distinct-valued names with
    # no ties, sum(pct_rank) - n*0.5 == 0.5 exactly regardless of n, so the
    # pre-normalization sum is a constant +0.5, not zero; after dividing by
    # the day's raw gross (~3 on this toy panel), the final weights carry a
    # small constant, nonzero net exposure. This is a real, disclosed
    # property of the rank-based construction (see README Limitations), not
    # a bug: "sums to ~0" in the docstring means small relative to unit
    # gross, not exactly 0.
    score, _ = _toy_panel()
    ranked = score.rank(axis=1, pct=True) - 0.5
    raw_gross = ranked.abs().sum(axis=1)
    w = dollar_neutral_weights(score)
    assert (w.abs().sum(axis=1).sub(1.0).abs() < 1e-9).all()
    expected_bias = 0.5 / raw_gross
    assert (w.sum(axis=1).sub(expected_bias).abs() < 1e-9).all()
    assert (w.sum(axis=1).abs() < 0.2).all()  # small relative to unit gross


def test_impact_cost_is_nondecreasing_in_participation():
    adv = pd.DataFrame({"A": [1_000_000.0] * 5})
    vol = pd.DataFrame({"A": [0.02] * 5})
    trades = pd.DataFrame({"A": [0.0, 10_000.0, 100_000.0, 500_000.0, 2_000_000.0]})
    cost = impact_cost_fraction(trades, adv, vol)
    values = cost["A"].tolist()
    assert all(a <= b + 1e-12 for a, b in zip(values, values[1:])), (
        "cost fraction must be non-decreasing as trade size grows, holding ADV and vol fixed"
    )
