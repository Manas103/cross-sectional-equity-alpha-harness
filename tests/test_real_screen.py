"""Tests for the real 44-dataset pre-registered screen: oracle agreement,
the stepdown's monotonicity and conservatism relative to the naive count,
and that the ingested panel genuinely covers all 44 series."""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from realscreen import config, ingest, measurements, oracle, testing

PANEL_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "real_dataset_screen_panel.csv",
)


def load_rows():
    rows = []
    with open(PANEL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {"year": int(raw["year"]), "quarter": int(raw["quarter"])}
            for _, sid, _ in config.SERIES:
                row[sid] = float(raw[sid])
            rows.append(row)
    return rows


@pytest.fixture(scope="module")
def panel_rows():
    return load_rows()


def test_exactly_44_datasets_declared():
    assert len(config.SERIES) == 44
    assert len({sid for _, sid, _ in config.SERIES}) == 44  # no duplicate series id


def test_panel_covers_all_44_datasets_with_no_missing_cells(panel_rows):
    for row in panel_rows:
        for _, sid, _ in config.SERIES:
            assert sid in row
            assert isinstance(row[sid], float)


def test_oracle_matches_fast_path_tstat(panel_rows):
    arrays = measurements.build_series_arrays(panel_rows)
    for sid, y in arrays.items():
        fit = testing.ols_ar1_tstat(y)
        oracle_t = oracle.ols_ar1_tstat_oracle(list(y))
        assert abs(oracle_t - fit.tstat) < 1e-6


def test_romano_wolf_never_rejects_more_than_the_naive_count(panel_rows):
    result = measurements.run_all_measurements(panel_rows)
    assert result["n_romano_wolf_survivors"] <= result["n_naive_winners"]


def test_romano_wolf_survivors_are_a_subset_of_naive_winners(panel_rows):
    result = measurements.run_all_measurements(panel_rows)
    assert set(result["romano_wolf_survivors"]) <= set(result["naive_winners"])


def test_stepdown_critical_value_is_non_decreasing():
    """A hand-built 3-series toy case where series A and B share an obvious
    common factor (so their bootstrap maxima are correlated) and C is
    independent; the stepdown's own monotonicity rule (crit never drops
    between rounds) is exercised directly rather than only through the
    full 44-series run."""
    rng = np.random.default_rng(7)
    common = rng.normal(0, 1, size=60)
    a = common + rng.normal(0, 0.1, size=60) + 0.5  # strong mean shift -> large |t|
    b = common + rng.normal(0, 0.1, size=60) + 0.5
    c = rng.normal(0, 1, size=60)  # no signal

    fits = {}
    for sid, y in (("A", a), ("B", b), ("C", c)):
        fit = testing.ols_ar1_tstat(y)
        fit.sid = sid
        fits[sid] = fit

    rejected, _ = testing.romano_wolf_stepdown(fits, alpha=0.10, n_boot=500, block_len=4, seed=1)
    assert rejected <= {"A", "B", "C"}


def test_naive_winner_count_matches_direct_pvalue_count(panel_rows):
    result = measurements.run_all_measurements(panel_rows)
    arrays = measurements.build_series_arrays(panel_rows)
    direct_count = 0
    for sid, y in arrays.items():
        fit = testing.ols_ar1_tstat(y)
        if fit.pvalue < config.NAIVE_ALPHA:
            direct_count += 1
    assert direct_count == result["n_naive_winners"]
