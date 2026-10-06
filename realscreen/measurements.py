"""Runs the pre-registered specification over all 44 real series and
reports the naive and Romano-Wolf-corrected winner counts.
"""

from __future__ import annotations

import numpy as np

from realscreen import config, testing


def build_series_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    arrays = {}
    for _, sid, _ in config.SERIES:
        arrays[sid] = np.array([r[sid] for r in rows], dtype=float)
    return arrays


def run_all_measurements(rows: list[dict]) -> dict:
    arrays = build_series_arrays(rows)
    fits = {}
    for sid, y in arrays.items():
        fit = testing.ols_ar1_tstat(y)
        fit.sid = sid
        fits[sid] = fit

    naive = testing.naive_winners(fits, config.NAIVE_ALPHA)
    rejected, boot_t = testing.romano_wolf_stepdown(
        fits, config.FWER_ALPHA, config.N_BOOT, config.BLOCK_LEN, config.SEED
    )

    return {
        "n_datasets": len(config.SERIES),
        "n_quarters": len(rows),
        "naive_alpha": config.NAIVE_ALPHA,
        "fwer_alpha": config.FWER_ALPHA,
        "naive_winners": naive,
        "n_naive_winners": len(naive),
        "romano_wolf_survivors": sorted(rejected),
        "n_romano_wolf_survivors": len(rejected),
        "fits": fits,
    }
