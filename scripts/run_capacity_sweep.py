"""Builds a cross-sectional long-short portfolio from the signals that
survived run_signal_mining.py, then simulates trading it at increasing
gross dollar size under a square-root market-impact cost model, to find
the size at which net Sharpe first falls below 0.5.

Usage: python scripts/run_signal_mining.py   (must run first)
       python scripts/run_capacity_sweep.py
Appends to: docs/benchmark_output.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xsalpha import ic, impact, portfolio
from xsalpha.neutralize import sector_neutralize
from xsalpha.signals import build_signals
from xsalpha.simulate import simulate_panel
from xsalpha.splits import make_purged_folds

SIZES = [1e6, 5e6, 1e7, 2.5e7, 5e7, 7.5e7, 1e8, 1.25e8, 1.5e8, 1.8e8,
          2e8, 2.5e8, 3e8, 4e8, 5e8, 7.5e8, 1e9, 1.5e9, 2e9,
          # Extended (second genuine attempt): the first sweep never
          # crossed net Sharpe 0.5 within $2B, because the 3 surviving
          # signals carry an unusually strong, persistent IC (~0.045) on
          # this synthetic panel; widen the range rather than the
          # threshold or the cost coefficient to find where it actually
          # crosses, if it does within a plausible size range.
          3e9, 5e9, 7.5e9, 1e10, 1.5e10, 2e10, 3e10, 5e10, 7.5e10, 1e11]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    docs_dir = repo_root / "docs"

    with open(data_dir / "survivors.json") as f:
        surv_info = json.load(f)
    survivors = surv_info["survivors"]

    lines = []
    lines.append("")
    lines.append("Capacity sweep (square-root market-impact cost model)")
    lines.append(f"Survivor signals feeding the portfolio: {len(survivors)}: {survivors}")

    if not survivors:
        lines.append("No signals survived the deflated-Sharpe correction; capacity sweep not applicable.")
        with open(docs_dir / "benchmark_output.txt", "a") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        return

    panel = simulate_panel()
    n_days = len(panel["dates"])
    folds = make_purged_folds(n_days)
    test_dates = panel["dates"][sorted({d for fold in folds for d in fold.test_idx})]

    all_signals = build_signals()
    survivor_scores = {name: all_signals[name](panel) for name in survivors}
    weights = portfolio.build_survivor_portfolio(panel, survivor_scores, test_dates)

    fwd_ret_1d = ic.forward_return(panel["close"], horizon=1)

    db_path = str(data_dir / "panel.duckdb")
    sweep = impact.capacity_sweep(panel, weights, fwd_ret_1d, SIZES, db_path=db_path)
    crossing = impact.find_capacity_crossing(sweep, threshold=0.5)

    lines.append(f"{'gross_$':>14s} {'gross_sharpe':>13s} {'net_sharpe':>11s} {'cost_bps':>9s}")
    for _, row in sweep.iterrows():
        lines.append(f"{row['gross_dollars']:14,.0f} {row['gross_sharpe']:13.3f} "
                      f"{row['net_sharpe']:11.3f} {row['mean_daily_cost_bps_of_gross']:9.2f}")
    lines.append("")
    if crossing == float("inf"):
        lines.append("net Sharpe never crossed below 0.5 within the sweep range")
    else:
        lines.append(f"CAPACITY: net Sharpe crosses below 0.5 at gross size ${crossing:,.0f}")
        lines.append(f"target from the resume was $180,000,000; "
                      f"measured crossing is {'above' if crossing > 1.8e8 else 'below'} that target "
                      f"by a factor of {crossing / 1.8e8:.2f}x")

    out = "\n".join(lines)
    print(out)
    with open(docs_dir / "benchmark_output.txt", "a") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
