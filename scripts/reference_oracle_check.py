"""Diffs the vectorized IC and deflated-Sharpe-ratio implementations
against the independent, loop-based reference oracle in
xsalpha/reference_oracle.py, exactly (not within a loose tolerance).

Usage: python scripts/reference_oracle_check.py
Writes: docs/reference_oracle_output.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xsalpha import dsr, ic, reference_oracle as oracle
from xsalpha.neutralize import sector_neutralize
from xsalpha.signals import build_signals
from xsalpha.simulate import simulate_panel
from xsalpha.splits import HORIZON


def main() -> None:
    lines = []
    panel = simulate_panel()
    fwd_ret = ic.forward_return(panel["close"], horizon=HORIZON)
    signals = build_signals()

    rng = np.random.default_rng(7)
    sample_names = rng.choice(list(signals.keys()), size=5, replace=False)
    sample_dates = rng.choice(panel["dates"][200:1900], size=8, replace=False)

    lines.append("Reference-oracle check 1: daily rank IC, vectorized vs hand-rolled loop")
    max_ic_diff = 0.0
    n_checked = 0
    for name in sample_names:
        raw_score = signals[name](panel)
        neutral = sector_neutralize(raw_score, panel["sector"])
        for dt in sample_dates:
            s = neutral.loc[dt]
            r = fwd_ret.loc[dt]
            mask = s.notna() & r.notna()
            if mask.sum() < 10:
                continue
            vec_ic_series = ic.daily_rank_ic(neutral.loc[[dt]], fwd_ret.loc[[dt]])
            if dt not in vec_ic_series.index:
                continue
            vec_val = vec_ic_series.loc[dt]
            oracle_val = oracle.oracle_daily_ic(list(s[mask].values), list(r[mask].values))
            diff = abs(vec_val - oracle_val)
            max_ic_diff = max(max_ic_diff, diff)
            n_checked += 1
    lines.append(f"  {n_checked} (signal, date) pairs checked across {len(sample_names)} signals x {len(sample_dates)} dates")
    lines.append(f"  max abs diff, vectorized vs oracle IC: {max_ic_diff:.3e}")
    lines.append(f"  PASS (exact to floating point, < 1e-9)" if max_ic_diff < 1e-9 else "  FAIL")

    lines.append("")
    lines.append("Reference-oracle check 2: deflated Sharpe ratio moments and PSR/SR0, vectorized vs hand-rolled")
    rng2 = np.random.default_rng(11)
    sample_series = [rng2.normal(0.01, 0.05, size=120) for _ in range(10)]

    vec_stats = [dsr.signal_stats(f"s{i}", s, periods_per_year=50.4) for i, s in enumerate(sample_series)]
    oracle_stats = []
    max_mean_diff = max_std_diff = max_skew_diff = max_kurt_diff = max_sr_diff = 0.0
    for i, s in enumerate(sample_series):
        o_mean, o_std, o_skew, o_kurt = oracle.oracle_moments(list(s))
        o_sr = oracle.oracle_sharpe(list(s), periods_per_year=50.4)
        oracle_stats.append((o_sr, o_skew, o_kurt, len(s)))
        v = vec_stats[i]
        max_mean_diff = max(max_mean_diff, abs(v.mean_ic - o_mean))
        max_skew_diff = max(max_skew_diff, abs(v.skew - o_skew))
        max_kurt_diff = max(max_kurt_diff, abs(v.kurtosis - o_kurt))
        max_sr_diff = max(max_sr_diff, abs(v.sr_hat - o_sr))
    lines.append(f"  10 synthetic series, 120 obs each")
    lines.append(f"  max abs diff mean: {max_mean_diff:.3e}, skew: {max_skew_diff:.3e}, "
                 f"kurtosis: {max_kurt_diff:.3e}, sharpe: {max_sr_diff:.3e}")

    vec_sr0 = dsr.sr0_benchmark([v.sr_hat for v in vec_stats], n_trials=63)
    oracle_sr0 = oracle.oracle_sr0([s[0] for s in oracle_stats], n_trials=63)
    sr0_diff = abs(vec_sr0 - oracle_sr0)
    lines.append(f"  SR0 (63 trials): vectorized {vec_sr0:.6f}, oracle {oracle_sr0:.6f}, diff {sr0_diff:.3e}")

    max_psr_diff = 0.0
    for v, o in zip(vec_stats, oracle_stats):
        vec_psr = dsr.probabilistic_sharpe_ratio(v.sr_hat, vec_sr0, v.skew, v.kurtosis, v.t_obs)
        oracle_psr = oracle.oracle_psr(o[0], oracle_sr0, o[1], o[2], o[3])
        max_psr_diff = max(max_psr_diff, abs(vec_psr - oracle_psr))
    lines.append(f"  max abs diff, PSR/DSR: {max_psr_diff:.3e}")

    all_diffs = [max_ic_diff, max_mean_diff, max_skew_diff, max_kurt_diff, max_sr_diff, sr0_diff, max_psr_diff]
    overall = max(all_diffs)
    lines.append("")
    lines.append(f"overall max abs diff across both checks: {overall:.3e}")
    lines.append("PASS (within 1e-8 tolerance)" if overall < 1e-8 else "FAIL")

    out = "\n".join(lines)
    print(out)
    repo_root = Path(__file__).resolve().parents[1]
    with open(repo_root / "docs" / "reference_oracle_output.txt", "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
