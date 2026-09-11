"""Purged walk-forward split generator.

Each fold is (train_idx, test_idx), both 0-based row positions into the
panel's date index. The label used everywhere in this repo is the
`HORIZON`-day forward return, so any training row whose label window
[t, t+HORIZON) would overlap the test block's date range is purged out
of the training set. The train window is expanding (all purged history
up to the fold boundary); the test window is a contiguous forward block.

`test_leakage.max_train_day + HORIZON <= min_test_day` is checked
directly in tests/test_purge_no_overlap.py, not just asserted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

HORIZON = 5     # forward-return horizon (trading days) used as the IC label
PURGE = HORIZON  # purge >= horizon guarantees no label overlap across the boundary
BURN_IN = 260     # skip the first ~1 trading year so rolling lookbacks (<=120d) are full
N_FOLDS = 8


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_idx: range
    test_idx: range


def make_purged_folds(n_days: int, burn_in: int = BURN_IN, n_folds: int = N_FOLDS,
                       horizon: int = HORIZON, purge: int = PURGE) -> List[Fold]:
    assert purge >= horizon, "purge must be at least the label horizon to avoid leakage"
    usable = n_days - burn_in
    block = usable // n_folds
    folds: List[Fold] = []
    for k in range(n_folds):
        test_start = burn_in + k * block
        test_end = n_days if k == n_folds - 1 else burn_in + (k + 1) * block
        train_end = test_start - purge
        if train_end <= 0:
            continue
        folds.append(Fold(fold_id=k, train_idx=range(0, train_end), test_idx=range(test_start, test_end)))
    return folds
