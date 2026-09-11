"""63 formulaic cross-sectional signals, in the spirit of WorldQuant's
"101 Formulaic Alphas": short expressions over rank/delay/delta/ts_*/
correlation/decay_linear operators applied to price, volume and two
derived fundamental-like fields (`earn_yield`, `quality`).

These are NOT the literal published 101 alphas; they are 11 short
formula templates, each instantiated over a fixed, pre-specified set of
lookback windows, for exactly 63 signals total. The template/window
choices below were fixed before any signal was scored (see
`scripts/run_signal_mining.py` and the README Findings section); nothing
here was added or removed after seeing which ones survived.

Two templates (D: value/earnings-yield, E: volume surprise) and the
momentum/low-vol templates (B, C) are built over fields that
`xsalpha/simulate.py` deliberately makes weakly informative. The other
seven templates (A, F, G, H, I, J, K) are built the same way over fields
that are either pure short-term reversal (A, which is genuinely a small
real effect in the simulator) or over the `quality` decoy field and
generic price/volume ratios with no embedded predictive relationship.
Which formulas actually survive the deflated-Sharpe correction is a
measured result, not a guaranteed one; see signals.md and the README.
"""
from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

from . import operators as op

SignalFn = Callable[[dict], pd.DataFrame]


def _ret1(panel: dict) -> pd.DataFrame:
    close = panel["close"]
    return close / close.shift(1) - 1.0


def build_signals() -> Dict[str, SignalFn]:
    signals: Dict[str, SignalFn] = {}

    # A. short-term reversal (5)
    for n in (1, 2, 3, 5, 8):
        def f(panel, n=n):
            return op.rank(-1 * op.delta(panel["close"], n))
        signals[f"revshort_{n}"] = f

    # B. momentum (6)
    for n in (10, 20, 40, 60, 90, 120):
        def f(panel, n=n):
            return op.rank(op.ts_sum(_ret1(panel), n))
        signals[f"mom_{n}"] = f

    # C. low volatility (5)
    for n in (10, 20, 40, 60, 90):
        def f(panel, n=n):
            return op.rank(-1 * op.ts_std(_ret1(panel), n))
        signals[f"lowvol_{n}"] = f

    # D. value / earnings-yield (6)
    signals["ey_level"] = lambda panel: op.rank(panel["earn_yield"])
    signals["ey_smooth_20"] = lambda panel: op.rank(op.ts_mean(panel["earn_yield"], 20))
    signals["ey_smooth_60"] = lambda panel: op.rank(op.ts_mean(panel["earn_yield"], 60))
    signals["ey_delta_20"] = lambda panel: op.rank(op.delta(panel["earn_yield"], 20))
    signals["ey_delta_60"] = lambda panel: op.rank(op.delta(panel["earn_yield"], 60))
    signals["ey_tsrank_60"] = lambda panel: op.rank(op.ts_rank(panel["earn_yield"], 60))

    # E. volume surprise (6)
    for n in (5, 10, 20, 40, 60, 90):
        def f(panel, n=n):
            return op.rank(panel["volume"] / op.ts_mean(panel["volume"], n))
        signals[f"volsurprise_{n}"] = f

    # F. quality decoy (6), structurally identical to D but on the
    # uninformative field
    signals["quality_level"] = lambda panel: op.rank(panel["quality"])
    signals["quality_smooth_20"] = lambda panel: op.rank(op.ts_mean(panel["quality"], 20))
    signals["quality_smooth_60"] = lambda panel: op.rank(op.ts_mean(panel["quality"], 60))
    signals["quality_delta_20"] = lambda panel: op.rank(op.delta(panel["quality"], 20))
    signals["quality_delta_60"] = lambda panel: op.rank(op.delta(panel["quality"], 60))
    signals["quality_tsrank_60"] = lambda panel: op.rank(op.ts_rank(panel["quality"], 60))

    # G. volume/close correlation decoy (6)
    for n in (5, 10, 20, 40, 60, 90):
        def f(panel, n=n):
            return op.rank(op.correlation(panel["volume"], panel["close"], n))
        signals[f"corr_vol_close_{n}"] = f

    # H. intraday range decoy (6)
    for n in (5, 10, 20, 40, 60, 90):
        def f(panel, n=n):
            rng = (panel["high"] - panel["low"]) / panel["close"]
            return op.rank(op.ts_mean(rng, n))
        signals[f"range_{n}"] = f

    # I. vwap deviation decoy (6)
    for n in (5, 10, 20, 40, 60, 90):
        def f(panel, n=n):
            dev = (panel["close"] - panel["vwap"]) / panel["vwap"]
            return op.rank(op.ts_mean(dev, n))
        signals[f"vwapdev_{n}"] = f

    # J. time-since-high decoy (5)
    for n in (10, 20, 40, 60, 90):
        def f(panel, n=n):
            return op.rank(-1 * op.ts_argmax(panel["close"], n))
        signals[f"argmaxclose_{n}"] = f

    # K. decay-weighted return decoy (6)
    for n in (5, 10, 20, 40, 60, 90):
        def f(panel, n=n):
            return op.rank(op.decay_linear(_ret1(panel), n))
        signals[f"decaymom_{n}"] = f

    assert len(signals) == 63, f"expected 63 signals, built {len(signals)}"
    return signals
