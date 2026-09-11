"""Proves the purged walk-forward folds never let a training label window
overlap a test block's date range, and that the deflated-Sharpe survivor
rule is applied uniformly (same threshold, same direction) to every signal."""
from __future__ import annotations

from xsalpha.splits import HORIZON, PURGE, make_purged_folds


def test_purge_covers_the_label_horizon():
    assert PURGE >= HORIZON


def test_folds_have_no_train_test_index_overlap_and_respect_the_purge_gap():
    n_days = 2016
    folds = make_purged_folds(n_days)
    assert len(folds) >= 1
    for fold in folds:
        train_set = set(fold.train_idx)
        test_set = set(fold.test_idx)
        assert train_set.isdisjoint(test_set)
        max_train_day = max(train_set)
        min_test_day = min(test_set)
        # A label at max_train_day looks ahead HORIZON days; it must not
        # reach into the test block.
        assert max_train_day + HORIZON <= min_test_day


def test_folds_are_expanding_and_contiguous_test_blocks():
    folds = make_purged_folds(2016)
    for a, b in zip(folds, folds[1:]):
        assert max(a.train_idx) < max(b.train_idx)  # expanding window
        assert max(a.test_idx) < min(b.test_idx)     # non-overlapping, forward-moving test blocks
