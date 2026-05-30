"""Purged & Embargoed K-Fold cross-validation.

Reference
---------
López de Prado (2018), *Advances in Financial Machine Learning*,
§7.4 "Purged K-Fold CV with Embargo".

Problem
-------
Naïve K-Fold CV on financial time series leaks information: returns or labels
that span multiple periods cause overlap between train and test sets. Random
shuffling makes the leak worse. Two corrections are needed:

1. **Purge.** Drop training observations whose evaluation horizon overlaps the
   test set. If labels are forward-looking by `label_horizon` periods, any
   training observation at index `t` with `t + label_horizon ≥ test_start`
   leaks the test label into training. Such observations are purged.

2. **Embargo.** Even after the test set ends, sequential correlations may
   still let the model "see" test outcomes via near-future observations. Skip
   the next `embargo` observations after the test fold from training.

This module produces purely index-based folds — caller decides how to map
indices to features/labels. Splits are *contiguous and ordered* in time
(`shuffle=False` always), which is the only correct mode for time-series.

Determinism
-----------
Given identical `n_obs`, `n_folds`, `label_horizon`, `embargo`, the returned
folds are bit-exactly reproducible. No RNG involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PurgedFold:
    """One fold of a purged & embargoed K-Fold split.

    Attributes
    ----------
    fold_index:
        Zero-based index in the K-fold sequence.
    test_indices:
        Contiguous array of indices for the test set (sorted ascending).
    train_indices:
        Sorted indices for the training set after purge + embargo. May be
        smaller than `n_obs − len(test_indices)` because of removed leakage
        zones.
    purged_count:
        How many indices were dropped by the purge rule.
    embargoed_count:
        How many indices were dropped by the embargo rule.
    """

    fold_index: int
    test_indices: np.ndarray
    train_indices: np.ndarray
    purged_count: int
    embargoed_count: int


def purged_kfold(
    n_obs: int,
    *,
    n_folds: int,
    label_horizon: int = 0,
    embargo: int = 0,
) -> list[PurgedFold]:
    """Generate `n_folds` purged + embargoed cross-validation splits.

    Parameters
    ----------
    n_obs:
        Total number of observations (T).
    n_folds:
        K, the number of folds. Must satisfy 2 ≤ K ≤ T.
    label_horizon:
        Forward-looking horizon h of the prediction target, in observations.
        Indices in `[test_start − h, test_start)` are purged.
    embargo:
        Number of observations to embargo after each test fold.

    Returns
    -------
    list of :class:`PurgedFold`, in chronological order.
    """
    if n_obs < 2:
        raise ValueError("n_obs must be ≥ 2")
    if not (2 <= n_folds <= n_obs):
        raise ValueError(f"n_folds must be in [2, {n_obs}], got {n_folds}")
    if label_horizon < 0 or embargo < 0:
        raise ValueError("label_horizon and embargo must be non-negative")

    all_idx = np.arange(n_obs)
    fold_edges = np.linspace(0, n_obs, n_folds + 1, dtype=int)

    folds: list[PurgedFold] = []
    for k in range(n_folds):
        test_start = int(fold_edges[k])
        test_end = int(fold_edges[k + 1])  # exclusive
        test_idx = all_idx[test_start:test_end]

        # Build the leakage mask in O(T): mark indices that must be excluded
        excluded = np.zeros(n_obs, dtype=bool)
        excluded[test_start:test_end] = True

        purge_lo = max(test_start - label_horizon, 0)
        purged_mask = np.zeros(n_obs, dtype=bool)
        if label_horizon > 0:
            purged_mask[purge_lo:test_start] = True

        embargo_hi = min(test_end + embargo, n_obs)
        embargoed_mask = np.zeros(n_obs, dtype=bool)
        if embargo > 0:
            embargoed_mask[test_end:embargo_hi] = True

        excluded |= purged_mask | embargoed_mask
        train_idx = all_idx[~excluded]

        folds.append(
            PurgedFold(
                fold_index=k,
                test_indices=test_idx,
                train_indices=train_idx,
                purged_count=int(purged_mask.sum()),
                embargoed_count=int(embargoed_mask.sum()),
            )
        )

    return folds
