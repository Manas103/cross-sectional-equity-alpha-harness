"""Synthetic 500-name, 8-year daily equity panel with real cross-sectional
structure and a small number of genuinely informative latent ingredients.

This is NOT real market data. See README.md "Honest framing" for why.

Generative model, day by day, causal (nothing at day t uses information
only available after day t):

  total_return[i, t] = beta_mkt[i]  * mkt_ret[t]
                      + beta_sec[i]  * sector_ret[sector[i], t]
                      + idio[i, t]                      (mild AR(1), see below)
                      + w_mom * xsec_zscore(mom20[:, t-1])[i]
                      + w_vol * xsec_zscore(-vol20[:, t-1])[i]
                      + w_ey  * xsec_zscore(earn_yield[:, t])[i]
                      + w_vp  * smart_money[i, t-1]

idio[i, t] = -rho_rev * idio[i, t-1] + eps[i, t] gives a small negative
lag-1 autocorrelation (the textbook short-term reversal effect) without
adding a separate ingredient column for it.

The four additive ingredients (momentum, low-vol, value/earnings-yield,
volume-leads-price) are individually small (each contributes roughly
5-15% of one day's idiosyncratic return standard deviation) so that no
single formulaic signal trivially reproduces them; a signal-mining sweep
has to actually find them against 8 years of noise, exactly like the
resume claim describes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_STOCKS = 500
N_DAYS = 2016  # ~8 trading years at 252 days/year
N_SECTORS = 10
SECTOR_NAMES = [
    "Technology", "Financials", "Health Care", "Consumer Discretionary",
    "Consumer Staples", "Industrials", "Energy", "Materials",
    "Utilities", "Real Estate",
]

# ingredient strengths, chosen once before any signal was mined, not tuned
# after seeing IC results (see README "Findings" for the attempt log)
W_MOM = 0.00028
W_VOL = 0.00022
W_EY = 0.00030
W_VP = 0.00035
RHO_REV = 0.045

MKT_DAILY_VOL = 0.009
MKT_DAILY_DRIFT = 0.00028
SECTOR_IDIO_VOL = 0.006
SECTOR_MKT_BETA = 0.7


def _xsec_zscore(row: np.ndarray) -> np.ndarray:
    mu = row.mean()
    sd = row.std()
    if sd < 1e-12:
        return np.zeros_like(row)
    return (row - mu) / sd


def simulate_panel(seed: int = 20260901) -> dict:
    """Returns a dict of DataFrames (index=trading dates, columns=tickers)
    for close, open, high, low, volume, vwap, earn_yield, quality, plus
    'sector' (Series ticker -> sector name) and 'tickers'."""
    rng = np.random.default_rng(seed)

    tickers = [f"SIM{i:04d}" for i in range(N_STOCKS)]
    sector_id = np.array([i % N_SECTORS for i in range(N_STOCKS)])
    rng.shuffle(sector_id)

    dates = pd.bdate_range(end="2024-12-31", periods=N_DAYS)

    beta_mkt = np.clip(rng.normal(1.0, 0.3, N_STOCKS), 0.2, 2.2)
    beta_sec = np.clip(rng.normal(1.0, 0.25, N_STOCKS), 0.1, 2.0)
    idio_vol = np.clip(rng.lognormal(mean=np.log(0.016), sigma=0.35, size=N_STOCKS), 0.006, 0.05)
    base_price = np.clip(rng.lognormal(mean=np.log(55.0), sigma=0.75, size=N_STOCKS), 5.0, 800.0)
    base_vol = np.clip(rng.lognormal(mean=np.log(1_200_000), sigma=1.0, size=N_STOCKS), 20_000, 5.0e7)

    # slow-moving fundamental-like "earnings yield" characteristic, one
    # cross-sectional draw per quarter (63 trading days), AR(1) across
    # quarters so a stock's value character persists but does drift
    n_quarters = N_DAYS // 63 + 2
    ey_level = np.zeros((n_quarters, N_STOCKS))
    ey_char = rng.normal(0.0, 1.0, N_STOCKS)  # each stock's long-run value tilt
    ey_level[0] = ey_char + rng.normal(0, 0.3, N_STOCKS)
    for q in range(1, n_quarters):
        ey_level[q] = 0.85 * ey_level[q - 1] + 0.15 * ey_char + rng.normal(0, 0.25, N_STOCKS)
    earn_yield = np.repeat(ey_level, 63, axis=0)[:N_DAYS]

    # decoy fundamental-like field, genuinely uninformative, a random walk
    quality = np.cumsum(rng.normal(0, 0.02, size=(N_DAYS, N_STOCKS)), axis=0)
    quality += rng.normal(0, 1.0, N_STOCKS)  # cross-sectional dispersion

    mkt_ret = MKT_DAILY_DRIFT + MKT_DAILY_VOL * rng.standard_normal(N_DAYS)

    sector_ret = np.zeros((N_DAYS, N_SECTORS))
    for s in range(N_SECTORS):
        sector_ret[:, s] = SECTOR_MKT_BETA * mkt_ret + SECTOR_IDIO_VOL * rng.standard_normal(N_DAYS)

    idio = np.zeros((N_DAYS, N_STOCKS))
    eps = rng.standard_normal((N_DAYS, N_STOCKS)) * idio_vol[None, :]
    idio[0] = eps[0]
    for t in range(1, N_DAYS):
        idio[t] = -RHO_REV * idio[t - 1] + eps[t]

    smart_money = rng.standard_normal((N_DAYS, N_STOCKS))  # today's private volume signal

    total_ret = np.zeros((N_DAYS, N_STOCKS))
    for t in range(N_DAYS):
        contrib = (
            beta_mkt * mkt_ret[t]
            + beta_sec * sector_ret[t, sector_id]
            + idio[t]
        )
        contrib = contrib + W_EY * _xsec_zscore(earn_yield[t])
        if t >= 21:
            mom20 = total_ret[t - 21 : t - 1].sum(axis=0)
            vol20 = total_ret[t - 21 : t - 1].std(axis=0)
            contrib = contrib + W_MOM * _xsec_zscore(mom20)
            contrib = contrib + W_VOL * _xsec_zscore(-vol20)
        if t >= 1:
            contrib = contrib + W_VP * smart_money[t - 1]
        total_ret[t] = contrib

    close = base_price[None, :] * np.exp(np.cumsum(np.log1p(total_ret), axis=0))
    prev_close = np.vstack([base_price[None, :], close[:-1]])

    intraday_range = np.abs(rng.standard_normal((N_DAYS, N_STOCKS))) * idio_vol[None, :] * 0.6
    open_ = prev_close * (1 + 0.15 * (close / prev_close - 1) + 0.15 * rng.standard_normal((N_DAYS, N_STOCKS)) * idio_vol[None, :])
    high = np.maximum(open_, close) * (1 + intraday_range)
    low = np.minimum(open_, close) * (1 - intraday_range)
    vwap = (open_ + high + low + close) / 4.0

    vol_shock = np.exp(0.5 * smart_money + rng.normal(0, 0.35, size=(N_DAYS, N_STOCKS)) - 0.5 * (0.5 ** 2 + 0.35 ** 2))
    volume = base_vol[None, :] * vol_shock
    volume = np.maximum(volume, 100.0)

    def _df(arr):
        return pd.DataFrame(arr, index=dates, columns=tickers)

    panel = {
        "close": _df(close),
        "open": _df(open_),
        "high": _df(high),
        "low": _df(low),
        "vwap": _df(vwap),
        "volume": _df(volume),
        "earn_yield": _df(earn_yield),
        "quality": _df(quality),
        "returns": _df(total_ret),
        "sector": pd.Series([SECTOR_NAMES[s] for s in sector_id], index=tickers, name="sector"),
        "tickers": tickers,
        "dates": dates,
    }
    return panel
