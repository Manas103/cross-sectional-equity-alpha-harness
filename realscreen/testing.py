"""The fast path: vectorized OLS AR(1) t-statistics, and a joint circular
block-bootstrap Romano-Wolf (2005) stepdown.

The bootstrap draws one common block-start path per replication and
applies it to every one of the 44 series' own null (beta=0) residuals,
rather than resampling each series independently. That single shared path
is what lets the resulting max-statistic distribution inherit the real
cross-sectional correlation between, say, WTI and Brent crude, instead of
treating all 44 tests as independent the way a Bonferroni correction
would. `oracle.py` recomputes the per-series t-statistics with a
hand-rolled loop, no numpy, diffed against `ols_ar1_tstat` here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SeriesFit:
    sid: str
    beta: float
    tstat: float
    pvalue: float
    null_mean: float
    null_resid: np.ndarray
    lagged_x: np.ndarray


def ols_ar1_tstat(y: np.ndarray) -> SeriesFit:
    """y_t ~ const + beta * y_{t-1}, OLS, classical (non-robust) t-stat on
    beta, two-sided p-value. Also returns the null (beta=0) residual series
    `null_resid = y_t - mean(y_t)`, the object the bootstrap resamples."""
    x = y[:-1]
    yt = y[1:]
    n = len(yt)
    X = np.column_stack([np.ones(n), x])
    beta_hat, *_ = np.linalg.lstsq(X, yt, rcond=None)
    resid = yt - X @ beta_hat
    dof = n - 2
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    tstat = float(beta_hat[1] / se) if se > 0 else 0.0
    pvalue = float(2 * (1 - stats.t.cdf(abs(tstat), dof)))
    null_mean = float(yt.mean())
    return SeriesFit(
        sid="",
        beta=float(beta_hat[1]),
        tstat=tstat,
        pvalue=pvalue,
        null_mean=null_mean,
        null_resid=yt - null_mean,
        lagged_x=x,
    )


def naive_winners(fits: dict[str, SeriesFit], alpha: float) -> list[str]:
    return sorted(sid for sid, fit in fits.items() if fit.pvalue < alpha)


def _block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    idx = []
    while len(idx) < n:
        start = int(rng.integers(0, n))
        idx.extend((start + k) % n for k in range(block_len))
    return np.array(idx[:n])


def _bootstrap_tstat(fit: SeriesFit, bidx: np.ndarray) -> float:
    n = len(fit.lagged_x)
    y_star = fit.null_mean + fit.null_resid[bidx]
    X = np.column_stack([np.ones(n), fit.lagged_x])
    beta_hat, *_ = np.linalg.lstsq(X, y_star, rcond=None)
    resid = y_star - X @ beta_hat
    dof = n - 2
    sigma2 = float(resid @ resid) / dof if dof > 0 else 0.0
    if sigma2 <= 0:
        return 0.0
    xtx_inv = np.linalg.inv(X.T @ X)
    se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    return float(beta_hat[1] / se) if se > 0 else 0.0


def romano_wolf_stepdown(
    fits: dict[str, SeriesFit],
    alpha: float,
    n_boot: int,
    block_len: int,
    seed: int,
) -> tuple[set[str], dict[str, np.ndarray]]:
    """Returns (rejected_sids, boot_tstats). `boot_tstats[sid]` is the
    length-n_boot array of that series' bootstrap t-statistics, kept so
    `measurements.py` can report the critical values actually used."""
    rng = np.random.default_rng(seed)
    sids = list(fits.keys())
    n = len(fits[sids[0]].lagged_x)

    boot_t = {sid: np.empty(n_boot) for sid in sids}
    for b in range(n_boot):
        bidx = _block_bootstrap_indices(n, block_len, rng)
        for sid in sids:
            boot_t[sid][b] = _bootstrap_tstat(fits[sid], bidx)

    remaining = set(sids)
    rejected: set[str] = set()
    prev_crit = 0.0
    while remaining:
        max_boot = np.max([np.abs(boot_t[sid]) for sid in remaining], axis=0)
        crit = float(np.quantile(max_boot, 1.0 - alpha))
        crit = max(crit, prev_crit)
        prev_crit = crit
        newly_rejected = {sid for sid in remaining if abs(fits[sid].tstat) > crit}
        if not newly_rejected:
            break
        rejected |= newly_rejected
        remaining -= newly_rejected

    return rejected, boot_t
