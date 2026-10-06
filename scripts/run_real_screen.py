"""Runs the pre-registered screen over data/real_dataset_screen_panel.csv,
prints the measurement JSON, and writes one one-page card per dataset to
docs/dataset_cards/<SID>.md.
"""

from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from realscreen import config, measurements, oracle, testing


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {"year": int(raw["year"]), "quarter": int(raw["quarter"])}
            for _, sid, _ in config.SERIES:
                row[sid] = float(raw[sid])
            rows.append(row)
    return rows


def write_dataset_cards(result: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    naive_set = set(result["naive_winners"])
    rw_set = set(result["romano_wolf_survivors"])
    category_by_sid = {sid: cat for cat, sid, _ in config.SERIES}
    name_by_sid = {sid: name for _, sid, name in config.SERIES}
    for sid, fit in result["fits"].items():
        lines = [
            f"# {sid}: {name_by_sid[sid]}",
            "",
            f"Category: {category_by_sid[sid]}",
            "",
            "## Pre-registered specification",
            "",
            "Quarter-over-quarter percent growth, OLS AR(1) (`growth_t ~ const + "
            "beta * growth_{t-1}`), two-sided test of `beta = 0` against the "
            "random-walk null, identical for all 44 datasets in this screen.",
            "",
            "## Result",
            "",
            f"- t-statistic: **{fit.tstat:.4f}**",
            f"- beta (AR(1) coefficient): {fit.beta:.4f}",
            f"- naive p-value: {fit.pvalue:.4f}",
            f"- naive winner at alpha={config.NAIVE_ALPHA}: **{'yes' if sid in naive_set else 'no'}**",
            f"- survives Romano-Wolf stepdown at FWER={config.FWER_ALPHA}: **{'yes' if sid in rw_set else 'no'}**",
        ]
        with open(os.path.join(out_dir, f"{sid}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def main() -> None:
    panel_path = sys.argv[1] if len(sys.argv) > 1 else "data/real_dataset_screen_panel.csv"
    rows = load_rows(panel_path)
    result = measurements.run_all_measurements(rows)

    oracle_max_abs_diff = 0.0
    for sid, fit in result["fits"].items():
        y = [r[sid] for r in rows]
        oracle_t = oracle.ols_ar1_tstat_oracle(y)
        oracle_max_abs_diff = max(oracle_max_abs_diff, abs(oracle_t - fit.tstat))

    write_dataset_cards(result, "docs/dataset_cards")

    printable = {
        "n_datasets": result["n_datasets"],
        "n_quarters": result["n_quarters"],
        "naive_alpha": result["naive_alpha"],
        "fwer_alpha": result["fwer_alpha"],
        "naive_winners": result["naive_winners"],
        "n_naive_winners": result["n_naive_winners"],
        "romano_wolf_survivors": result["romano_wolf_survivors"],
        "n_romano_wolf_survivors": result["n_romano_wolf_survivors"],
        "oracle_vs_fast_path_max_abs_tstat_diff": oracle_max_abs_diff,
    }
    json.dump(printable, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
