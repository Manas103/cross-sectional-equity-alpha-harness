"""Builds the vintage store once, pins a manifest (data vintage digest + code
digest + seed), computes the pinned statistic twice independently, and
checks the two runs are bit-identical.

Writes docs/pinned_reproducibility_output.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xsalpha.run_manifest import build_manifest, compute_pinned_statistic
from xsalpha.vintage_store import build_simulated_vintage_store

NUM_ENTITIES = 500
NUM_DAYS = 2016


def main() -> None:
    trading_days = pd.bdate_range("2017-01-03", periods=NUM_DAYS)
    store = build_simulated_vintage_store(NUM_ENTITIES, trading_days)
    as_of_date = trading_days[1500]

    manifest = build_manifest(store, seed=4242)

    run_1 = compute_pinned_statistic(store, manifest, as_of_date)

    store_2 = build_simulated_vintage_store(NUM_ENTITIES, trading_days)
    manifest_2 = build_manifest(store_2, seed=4242)
    run_2 = compute_pinned_statistic(store_2, manifest_2, as_of_date)

    lines = [
        f"manifest: {manifest.to_json()}",
        f"run_1: {run_1}",
        f"run_2: {run_2}",
    ]

    exact_match = run_1 == run_2
    lines.append(f"exact match across two independent runs of the same manifest: {exact_match}")

    different_seed_manifest = build_manifest(store, seed=777)
    run_different_seed = compute_pinned_statistic(store, different_seed_manifest, as_of_date)
    differs_with_different_seed = run_1["seeded_resample_mean"] != run_different_seed["seeded_resample_mean"]
    lines.append(f"a different seed changes the seeded resample mean: {differs_with_different_seed}")

    output = "\n".join(lines)
    print(output)

    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "pinned_reproducibility_output.txt").write_text(output + "\n", encoding="utf-8")

    assert exact_match, "pinned run did not reproduce exactly"
    assert differs_with_different_seed, "seed appears to not be wired into the pinned statistic"


if __name__ == "__main__":
    main()
