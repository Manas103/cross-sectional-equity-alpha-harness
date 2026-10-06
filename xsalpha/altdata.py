"""Two alternative-data signal proxies, built on the existing 500-name
panel so they can be combined directly with the 63 formulaic signals,
without touching xsalpha/simulate.py's return-generating process (doing
so would silently change every already-measured number in this repo).

Both proxies reuse the panel's one genuinely informative fundamental-like
field, `earn_yield`, through the same mechanism the sibling repositories
in this portfolio use for alternative data: a noisy, uneven-coverage
re-read of a real signal, rather than a clean one. This is a deliberate,
disclosed honesty choice: an alt-data signal that is a cleaner read of
`earn_yield` than `earn_yield` itself would not be a credible analogue of
a real alternative-data source, which is almost always a noisier,
partial-coverage proxy for the fundamental it is standing in for.

- `nowcast_signal`: the same coverage-dependent sampling-noise mechanism
  as point-in-time-activity-index's `nowcast/` extension (noise std
  shrinks with sqrt(coverage)), applied here to `earn_yield` instead of
  a consumer-spending panel's ticket/transaction growth.
- `filing_signal`: a noisier read of the *change* in `earn_yield`
  (a revenue-surprise-shaped quantity), reflecting that a text-based
  signal measured in Filing-Language Signal for Next-Quarter Revenue
  Surprise (AUC 0.57, barely above the 0.50 baseline) is a materially
  weaker edge than a direct fundamental read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED_COVERAGE = 20261006
SEED_NOWCAST_NOISE = 20261007
SEED_FILING_NOISE = 20261008

COVERAGE_LOG_LOW = -4.0
COVERAGE_LOG_HIGH = -0.2
PANEL_BASE_SIZE = 5_000.0
BASE_NOISE_NOWCAST = 1.4
BASE_NOISE_FILING = 3.0  # materially noisier: the filing-language proxy


def per_ticker_coverage(tickers: list[str], seed: int = SEED_COVERAGE) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_cov = rng.uniform(COVERAGE_LOG_LOW, COVERAGE_LOG_HIGH, size=len(tickers))
    return pd.Series(np.exp(log_cov), index=tickers, name="coverage")


def _quarterly_noise(n_days: int, n_stocks: int, seed: int, std: np.ndarray) -> np.ndarray:
    """Noise drawn once per ~63-trading-day quarter and held constant
    within it, repeated to daily frequency. `earn_yield` itself is
    already piecewise-constant per quarter the same way (see
    simulate.py); re-drawing fresh noise every day here instead would
    inject a daily-jitter artifact that has nothing to do with the
    quarterly cadence a real panel or filing update actually has, and
    would show up as spuriously high portfolio turnover."""
    rng = np.random.default_rng(seed)
    n_quarters = n_days // 63 + 2
    draws = rng.normal(0.0, 1.0, size=(n_quarters, n_stocks)) * std[None, :]
    return np.repeat(draws, 63, axis=0)[:n_days]


def nowcast_signal(panel: dict) -> pd.DataFrame:
    """A coverage-noisy re-read of `earn_yield`, ranked. Lower coverage
    names get more sampling noise added, exactly the nowcast/ mechanism;
    the noise is drawn once per quarter, matching earn_yield's own
    quarterly-update cadence."""
    earn_yield = panel["earn_yield"]
    coverage = per_ticker_coverage(panel["tickers"])
    noise_std = (BASE_NOISE_NOWCAST / np.sqrt(coverage * PANEL_BASE_SIZE)).to_numpy()
    noise = _quarterly_noise(len(earn_yield), earn_yield.shape[1], SEED_NOWCAST_NOISE, noise_std)
    noisy = earn_yield + noise
    return noisy.rank(axis=1, pct=True)


def filing_signal(panel: dict) -> pd.DataFrame:
    """A noisier read of the change in `earn_yield` over the prior ~63
    trading days (one simulated quarter), ranked. Deliberately weaker
    than `nowcast_signal`: a text-derived proxy for a revenue surprise is
    a shakier read of the fundamental than a coverage-weighted panel
    re-read of its level. Noise is also drawn once per quarter."""
    earn_yield = panel["earn_yield"]
    d_ey = earn_yield - earn_yield.shift(63)
    coverage = per_ticker_coverage(panel["tickers"])
    noise_std = (BASE_NOISE_FILING / np.sqrt(coverage * PANEL_BASE_SIZE)).to_numpy()
    noise = _quarterly_noise(len(d_ey), d_ey.shape[1], SEED_FILING_NOISE, noise_std)
    noisy = d_ey + noise
    return noisy.rank(axis=1, pct=True)


def build_altdata_signals(panel: dict) -> dict[str, pd.DataFrame]:
    return {
        "altdata_nowcast": nowcast_signal(panel),
        "altdata_filing": filing_signal(panel),
    }
