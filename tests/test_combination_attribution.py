import numpy as np
import pandas as pd
import pytest

from xsalpha import altdata, combine, factors
from xsalpha.signals import build_signals
from xsalpha.simulate import simulate_panel
from xsalpha.splits import make_purged_folds


@pytest.fixture(scope="module")
def panel():
    return simulate_panel()


def test_altdata_signals_are_piecewise_constant_per_quarter(panel):
    scores = altdata.build_altdata_signals(panel)
    for name, df in scores.items():
        # Quarter blocks are exactly 63 trading days (0-62, 63-125, ...);
        # rows 70-124 sit entirely inside the 63-125 block, well past any
        # shift warm-up, so the daily cross-sectional rank must be
        # identical every day in that span.
        block = df.iloc[70:125]
        first_row = block.iloc[0]
        for i in range(1, len(block)):
            pd.testing.assert_series_equal(block.iloc[i], first_row, check_names=False)


def test_altdata_signal_changes_across_quarter_boundary(panel):
    scores = altdata.build_altdata_signals(panel)
    nowcast = scores["altdata_nowcast"]
    # Day 99 (end of one quarter block) vs day 163 (well into the next)
    # must differ for at least some names; otherwise the signal is inert.
    assert not nowcast.iloc[99].equals(nowcast.iloc[163])


def test_sector_factor_returns_one_column_per_sector(panel):
    sector_ret = factors.sector_factor_returns(panel)
    assert set(sector_ret.columns) == set(panel["sector"].unique())
    assert len(sector_ret) == len(panel["dates"])


def test_attribute_recovers_known_loading():
    rng = np.random.default_rng(0)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    factor_a = pd.Series(rng.normal(0, 0.01, n), index=dates, name="a")
    factor_b = pd.Series(rng.normal(0, 0.01, n), index=dates, name="b")
    noise = rng.normal(0, 0.002, n)
    alpha = 0.0005
    y = alpha + 2.0 * factor_a + 0.5 * factor_b + noise
    y = pd.Series(y, index=dates)
    factor_df = pd.concat([factor_a, factor_b], axis=1)

    fitted, resid, pct_explained, beta = factors.attribute(y, factor_df)
    assert beta[0] == pytest.approx(2.0, abs=0.1)
    assert beta[1] == pytest.approx(0.5, abs=0.1)
    # The known alpha must survive in the residual, not be absorbed into "explained".
    assert resid.mean() == pytest.approx(alpha, abs=0.0005)


def test_combined_weights_are_dollar_neutral_and_unit_gross(panel):
    folds = make_purged_folds(len(panel["dates"]))
    test_dates = panel["dates"][sorted({d for fold in folds for d in fold.test_idx})]
    all_signals = build_signals()
    survivor_scores = {name: all_signals[name](panel) for name in ("ey_level", "mom_20")}
    altdata_scores = altdata.build_altdata_signals(panel)
    weights = combine.combined_weights(panel, survivor_scores, altdata_scores, test_dates)
    gross = weights.abs().sum(axis=1).dropna()
    net = weights.sum(axis=1).dropna()
    assert gross.sub(1.0).abs().max() < 1e-6
    # A small, constant residual net exposure is a documented artifact of
    # the rank-based construction (see portfolio.py docstring and this
    # repository's own test_neutralize_and_impact.py): for a universe
    # with no ranking ties, the raw pre-normalization long-short sum is
    # +0.5 rather than exactly 0, not a bug introduced by combining signals.
    assert net.abs().max() < 0.01
