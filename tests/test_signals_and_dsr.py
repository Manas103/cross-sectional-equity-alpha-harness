"""Checks the signal set size and basic deflated-Sharpe-ratio sanity
properties (a benchmark built from more trials should not be easier to
clear, and PSR is bounded in [0, 1])."""
from __future__ import annotations

import numpy as np

from xsalpha.dsr import probabilistic_sharpe_ratio, signal_stats, sr0_benchmark
from xsalpha.signals import build_signals


def test_exactly_63_signals_built():
    signals = build_signals()
    assert len(signals) == 63


def test_sr0_benchmark_increases_with_more_trials_given_fixed_dispersion():
    rng = np.random.default_rng(3)
    sr_hats = list(rng.normal(1.0, 0.5, size=30))
    sr0_few = sr0_benchmark(sr_hats, n_trials=10)
    sr0_many = sr0_benchmark(sr_hats, n_trials=1000)
    assert sr0_many > sr0_few, "more trials should demand a higher Sharpe to clear by chance alone"


def test_psr_is_bounded_in_unit_interval():
    rng = np.random.default_rng(4)
    series = rng.normal(0.01, 0.05, size=200)
    stats = signal_stats("s", series, periods_per_year=50.4)
    psr = probabilistic_sharpe_ratio(stats.sr_hat, benchmark=0.0, skew=stats.skew,
                                      kurtosis=stats.kurtosis, t_obs=stats.t_obs)
    assert 0.0 <= psr <= 1.0
