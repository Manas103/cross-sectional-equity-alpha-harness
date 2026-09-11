"""Independent, deliberately slow, loop-based reference implementations
of (1) the daily rank information coefficient and (2) the deflated
Sharpe ratio moment/benchmark calculations, sharing no helper code with
`ic.py` or `dsr.py`. `scripts/reference_oracle_check.py` diffs these
against the vectorized versions exactly.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from scipy import stats


def oracle_rank(values: Sequence[float]) -> List[float]:
    """Average-rank (1-based, ties averaged), computed by explicit
    sorting and a linear scan, no numpy/pandas rank call."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def oracle_daily_ic(score_values: Sequence[float], fwd_values: Sequence[float]) -> float:
    """Spearman correlation for one day, hand-rolled: rank both series,
    then Pearson-correlate the ranks with an explicit summation loop."""
    assert len(score_values) == len(fwd_values)
    n = len(score_values)
    sr = oracle_rank(score_values)
    rr = oracle_rank(fwd_values)
    mean_sr = sum(sr) / n
    mean_rr = sum(rr) / n
    num = 0.0
    denom_s = 0.0
    denom_r = 0.0
    for i in range(n):
        ds = sr[i] - mean_sr
        dr = rr[i] - mean_rr
        num += ds * dr
        denom_s += ds * ds
        denom_r += dr * dr
    denom = math.sqrt(denom_s * denom_r)
    return num / denom if denom > 0 else float("nan")


def oracle_moments(values: Sequence[float]) -> Tuple[float, float, float, float]:
    """Returns (mean, std (ddof=1), skewness, kurtosis (non-excess, normal=3))
    computed with explicit summation loops, not scipy.stats/numpy."""
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    if std == 0 or n < 3:
        return mean, std, 0.0, 3.0
    m2 = sum((x - mean) ** 2 for x in values) / n
    m3 = sum((x - mean) ** 3 for x in values) / n
    m4 = sum((x - mean) ** 4 for x in values) / n
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2)
    return mean, std, skew, kurt


def oracle_sharpe(values: Sequence[float], periods_per_year: float) -> float:
    mean, std, _, _ = oracle_moments(values)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)


def oracle_sr0(sr_hats: Sequence[float], n_trials: int) -> float:
    n = len(sr_hats)
    mean = sum(sr_hats) / n
    var = sum((x - mean) ** 2 for x in sr_hats) / (n - 1) if n > 1 else 0.0
    sigma_sr = math.sqrt(var)
    if sigma_sr == 0.0 or n_trials <= 1:
        return 0.0
    gamma = 0.5772156649015329
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sigma_sr * ((1.0 - gamma) * z1 + gamma * z2)


def oracle_psr(sr_hat: float, benchmark: float, skew: float, kurtosis: float, t_obs: int) -> float:
    if t_obs <= 1:
        return 0.0
    denom = 1.0 - skew * sr_hat + ((kurtosis - 1.0) / 4.0) * sr_hat ** 2
    if denom <= 0:
        denom = 1e-12
    z = (sr_hat - benchmark) * math.sqrt(t_obs - 1) / math.sqrt(denom)
    return stats.norm.cdf(z)
