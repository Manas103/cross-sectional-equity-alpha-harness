"""Mines the 63 formulaic signals: sector-neutralizes each, scores its
out-of-sample rank IC under the purged walk-forward folds, applies the
deflated Sharpe ratio correction for 63 trials, and prints which
signals survive a pre-specified, mechanical threshold applied uniformly
to all 63 (never hand-picked after seeing results).

Usage: python scripts/run_signal_mining.py
Writes: docs/benchmark_output.txt (signal mining section),
        data/survivors.json (consumed by run_capacity_sweep.py)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xsalpha import db as xsdb
from xsalpha import dsr, ic
from xsalpha.neutralize import sector_neutralize
from xsalpha.signals import build_signals
from xsalpha.simulate import simulate_panel
from xsalpha.splits import HORIZON, make_purged_folds

DSR_THRESHOLD = 0.95  # pre-specified before any signal was scored


def main() -> None:
    t0 = time.time()
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "docs"
    data_dir = repo_root / "data"
    docs_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    panel = simulate_panel()
    n_days = len(panel["dates"])
    print(f"panel: {n_days} trading days x {len(panel['tickers'])} tickers, "
          f"{len(set(panel['sector'].values))} sectors")

    db_path = str(data_dir / "panel.duckdb")
    xsdb.build_duckdb(panel, db_path)
    print(f"panel written to DuckDB: {db_path}")

    folds = make_purged_folds(n_days)
    print(f"{len(folds)} purged walk-forward folds, horizon={HORIZON}, purge={HORIZON}")

    fwd_ret = ic.forward_return(panel["close"], horizon=HORIZON)
    signals = build_signals()
    print(f"{len(signals)} formulaic signals built")

    all_stats = []
    per_signal_signs = {}
    for name, fn in signals.items():
        raw_score = fn(panel)
        neutral = sector_neutralize(raw_score, panel["sector"])
        raw_ic = ic.daily_rank_ic(neutral, fwd_ret)
        oos_ic, fold_signs = ic.oos_ic_series(raw_ic, folds, panel["dates"], thin=HORIZON)
        periods_per_year = 252.0 / HORIZON
        stats = dsr.signal_stats(name, oos_ic.values, periods_per_year)
        all_stats.append(stats)
        per_signal_signs[name] = fold_signs
        print(f"  {name:20s} t_obs={stats.t_obs:4d} mean_ic={stats.mean_ic:+.4f} "
              f"sr_hat={stats.sr_hat:+.3f}")

    n_trials = len(all_stats)
    dsr_values = dsr.deflated_sharpe_ratios(all_stats, n_trials)
    sr0 = dsr.sr0_benchmark([s.sr_hat for s in all_stats], n_trials)

    survivors = []
    lines = []
    lines.append(f"Signal mining: {n_trials} formulaic signals, {n_days} trading days, "
                  f"{len(folds)} purged walk-forward folds (horizon={HORIZON}, purge={HORIZON})")
    lines.append(f"Deflated Sharpe benchmark SR0 (63 trials): {sr0:.4f}")
    lines.append(f"Survivor rule (pre-specified): DSR > {DSR_THRESHOLD} AND mean out-of-sample IC > 0")
    lines.append("")
    lines.append(f"{'signal':22s} {'mean_ic':>9s} {'sr_hat':>8s} {'skew':>7s} {'kurt':>7s} {'t_obs':>6s} {'dsr':>7s}  survives")
    for stats, dsr_val in sorted(zip(all_stats, dsr_values), key=lambda p: -p[1]):
        survives = (dsr_val > DSR_THRESHOLD) and (stats.mean_ic > 0)
        if survives:
            survivors.append(stats.name)
        lines.append(f"{stats.name:22s} {stats.mean_ic:+9.4f} {stats.sr_hat:+8.3f} "
                      f"{stats.skew:+7.3f} {stats.kurtosis:7.3f} {stats.t_obs:6d} {dsr_val:7.4f}  "
                      f"{'YES' if survives else 'no'}")
    lines.append("")
    lines.append(f"SURVIVORS: {len(survivors)} of {n_trials} signals: {survivors}")
    lines.append(f"elapsed: {time.time() - t0:.1f}s")

    out_text = "\n".join(lines)
    print("\n" + out_text)

    with open(docs_dir / "benchmark_output.txt", "w") as f:
        f.write(out_text + "\n")

    with open(data_dir / "survivors.json", "w") as f:
        json.dump({"survivors": survivors, "n_trials": n_trials, "sr0": sr0,
                   "dsr_threshold": DSR_THRESHOLD}, f, indent=2)


if __name__ == "__main__":
    main()
