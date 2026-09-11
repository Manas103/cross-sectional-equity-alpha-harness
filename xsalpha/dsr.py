"""Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014), correcting a
Sharpe-ratio estimate for having been chosen as the best (or evaluated
alongside) N independent trials, and for non-normal skew/kurtosis in the
underlying series.

    SR0     = expected maximum Sharpe ratio across N trials under the
              null of zero true skill, given the cross-trial dispersion
              sigma_SR of the N trials' own Sharpe estimates
    PSR(b)  = probability the true Sharpe exceeds benchmark b, given the
              observed Sharpe, its skew/kurtosis, and sample size T
    DSR     = PSR(SR0)

A signal is scored one at a time; SR0 is shared across all N=63 trials
because it depends only on N and on the cross-sectional dispersion of
the 63 trials' own Sharpe estimates (sigma_SR), not on any individual
trial's own skew/kurtosis/T.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


@dataclass
class SignalStats:
    name: str
    sr_hat: float   # annualized Sharpe of the out-of-sample IC series
    skew: float
    kurtosis: float  # non-excess (normal = 3.0)
    t_obs: int
    mean_ic: float


def signal_stats(name: str, ic_series: np.ndarray, periods_per_year: float) -> SignalStats:
    ic_series = np.asarray(ic_series, dtype=float)
    t_obs = len(ic_series)
    mean_ic = float(ic_series.mean())
    std_ic = float(ic_series.std(ddof=1)) if t_obs > 1 else 0.0
    sr_hat = (mean_ic / std_ic) * np.sqrt(periods_per_year) if std_ic > 0 else 0.0
    skew = float(stats.skew(ic_series)) if t_obs > 2 else 0.0
    kurt = float(stats.kurtosis(ic_series, fisher=False)) if t_obs > 3 else 3.0
    return SignalStats(name=name, sr_hat=sr_hat, skew=skew, kurtosis=kurt, t_obs=t_obs, mean_ic=mean_ic)


def sr0_benchmark(sr_hats: List[float], n_trials: int) -> float:
    """Expected maximum Sharpe ratio across n_trials independent trials
    under the null of zero skill, from the observed cross-trial
    dispersion of Sharpe estimates."""
    sigma_sr = float(np.std(np.asarray(sr_hats), ddof=1)) if len(sr_hats) > 1 else 0.0
    if sigma_sr == 0.0 or n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return sigma_sr * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def probabilistic_sharpe_ratio(sr_hat: float, benchmark: float, skew: float, kurtosis: float, t_obs: int) -> float:
    if t_obs <= 1:
        return 0.0
    denom = 1.0 - skew * sr_hat + ((kurtosis - 1.0) / 4.0) * sr_hat ** 2
    if denom <= 0:
        denom = 1e-12
    z = (sr_hat - benchmark) * np.sqrt(t_obs - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratios(stats_list: List[SignalStats], n_trials: int) -> List[float]:
    sr0 = sr0_benchmark([s.sr_hat for s in stats_list], n_trials)
    return [
        probabilistic_sharpe_ratio(s.sr_hat, sr0, s.skew, s.kurtosis, s.t_obs)
        for s in stats_list
    ]
