"""Combines the deflated-Sharpe survivors with the two alternative-data
signals, attributes the combined portfolio's daily return to sector,
size and momentum exposure, and prices the residual's Sharpe gross and
net of a half-spread plus square-root market impact at $250M.

Usage: python scripts/run_signal_mining.py   (must run first)
       python scripts/run_combination_attribution.py
Writes: docs/attribution_output.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xsalpha import altdata, combine
from xsalpha.signals import build_signals
from xsalpha.simulate import simulate_panel
from xsalpha.splits import make_purged_folds

GROSS_DOLLARS_FOR_COST = 250_000_000.0


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    docs_dir = repo_root / "docs"

    with open(data_dir / "survivors.json") as f:
        surv_info = json.load(f)
    survivors = surv_info["survivors"]

    panel = simulate_panel()
    n_days = len(panel["dates"])
    folds = make_purged_folds(n_days)
    test_dates = panel["dates"][sorted({d for fold in folds for d in fold.test_idx})]

    all_signals = build_signals()
    survivor_scores = {name: all_signals[name](panel) for name in survivors}
    altdata_scores = altdata.build_altdata_signals(panel)

    result = combine.run_attribution(
        panel, survivor_scores, altdata_scores, test_dates, GROSS_DOLLARS_FOR_COST
    )

    lines = []
    lines.append(
        f"Combination and factor attribution: {len(survivor_scores)} deflated-Sharpe "
        f"survivors ({list(survivor_scores)}) + {len(altdata_scores)} alternative-data "
        f"signals ({list(altdata_scores)}) = {len(survivor_scores) + len(altdata_scores)} "
        f"signals combined"
    )
    lines.append(f"combined portfolio evaluated over {len(result['gross_ret'])} test days")
    lines.append(f"combined gross Sharpe (unit gross exposure): {result['gross_sharpe_combined']:.3f}")
    lines.append(
        f"factor attribution (10 sector factors + size + momentum), return-based: "
        f"{result['r2_explained'] * 100:.1f}% of combined gross return explained"
    )
    lines.append(f"residual gross Sharpe: {result['gross_sharpe_residual']:.3f}")
    lines.append(
        f"residual net Sharpe at ${GROSS_DOLLARS_FOR_COST:,.0f} gross "
        f"(half-spread {combine.HALF_SPREAD_BPS * 1e4:.1f}bps + square-root impact): "
        f"{result['net_sharpe_residual']:.3f}"
    )
    lines.append(
        f"target from the resume: 62% explained, residual Sharpe 1.62 gross to 0.38 net"
    )

    out = "\n".join(lines)
    print(out)
    docs_dir.mkdir(exist_ok=True)
    with open(docs_dir / "attribution_output.txt", "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
