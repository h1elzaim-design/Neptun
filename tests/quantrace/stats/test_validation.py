"""Unit tests for quantrace.stats.validation (Purged & Embargoed K-Fold)."""

from __future__ import annotations

import numpy as np
import pytest

from quantrace.stats.validation import purged_kfold


class TestPurgedKfoldShape:
    def test_test_sets_cover_all_indices_no_overlap(self):
        n_obs, k = 1000, 5
        folds = purged_kfold(n_obs, n_folds=k, label_horizon=0, embargo=0)
        all_test = np.concatenate([f.test_indices for f in folds])
        assert all_test.size == n_obs
        assert np.array_equal(np.sort(all_test), np.arange(n_obs))

    def test_test_indices_are_contiguous_ranges(self):
        folds = purged_kfold(500, n_folds=4, label_horizon=10, embargo=5)
        for f in folds:
            diffs = np.diff(f.test_indices)
            assert (diffs == 1).all(), f"fold {f.fold_index} test indices not contiguous"

    def test_n_folds_count(self):
        for k in (2, 3, 5, 10):
            assert len(purged_kfold(200, n_folds=k)) == k


class TestPurgedKfoldLeakage:
    def test_no_train_test_overlap(self):
        folds = purged_kfold(1000, n_folds=5, label_horizon=20, embargo=10)
        for f in folds:
            overlap = np.intersect1d(f.train_indices, f.test_indices)
            assert overlap.size == 0, f"fold {f.fold_index} has overlap: {overlap}"

    def test_purge_zone_excluded(self):
        """An index at test_start - h .. test_start - 1 must not appear in training."""
        folds = purged_kfold(200, n_folds=4, label_horizon=15, embargo=0)
        for f in folds:
            test_start = int(f.test_indices[0])
            purge_zone = set(range(max(test_start - 15, 0), test_start))
            train_set = set(int(x) for x in f.train_indices)
            assert purge_zone.isdisjoint(train_set), (
                f"fold {f.fold_index} leaks purge zone into training"
            )

    def test_embargo_zone_excluded(self):
        folds = purged_kfold(200, n_folds=4, label_horizon=0, embargo=12)
        for f in folds:
            test_end = int(f.test_indices[-1]) + 1  # exclusive
            embargo_zone = set(range(test_end, min(test_end + 12, 200)))
            train_set = set(int(x) for x in f.train_indices)
            assert embargo_zone.isdisjoint(train_set), (
                f"fold {f.fold_index} leaks embargo zone into training"
            )

    def test_purged_and_embargoed_counts_reported(self):
        folds = purged_kfold(200, n_folds=4, label_horizon=8, embargo=5)
        # Middle folds get a full purge zone and a full embargo zone.
        middle = folds[1]
        assert middle.purged_count == 8
        assert middle.embargoed_count == 5
        # First fold has no purge zone before (test_start=0), so purge=0
        first = folds[0]
        assert first.purged_count == 0
        # Last fold has no embargo zone after, so embargo=0
        last = folds[-1]
        assert last.embargoed_count == 0


class TestPurgedKfoldDeterminism:
    def test_identical_inputs_yield_identical_folds(self):
        a = purged_kfold(500, n_folds=4, label_horizon=10, embargo=5)
        b = purged_kfold(500, n_folds=4, label_horizon=10, embargo=5)
        for fa, fb in zip(a, b, strict=False):
            assert np.array_equal(fa.test_indices, fb.test_indices)
            assert np.array_equal(fa.train_indices, fb.train_indices)


class TestPurgedKfoldValidation:
    def test_rejects_bad_n_folds(self):
        with pytest.raises(ValueError):
            purged_kfold(100, n_folds=1)
        with pytest.raises(ValueError):
            purged_kfold(100, n_folds=200)

    def test_rejects_negative_params(self):
        with pytest.raises(ValueError):
            purged_kfold(100, n_folds=3, label_horizon=-1)
        with pytest.raises(ValueError):
            purged_kfold(100, n_folds=3, embargo=-1)
