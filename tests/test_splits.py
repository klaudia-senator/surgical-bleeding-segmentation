from pathlib import Path

import pytest

from surgical_bleeding.splits import PairRecord, assert_no_group_overlap, make_folds


def records(count=12):
    return [PairRecord(Path(f"{i}.png"), Path(f"{i}.png"), f"video_{i // 4}") for i in range(count)]


def test_group_folds_do_not_mix_sources():
    values = records()
    for train_idx, val_idx in make_folds(values, n_splits=3, mode="group"):
        assert_no_group_overlap([values[i] for i in train_idx], [values[i] for i in val_idx])


def test_overlap_guard_rejects_leakage():
    values = records()
    with pytest.raises(ValueError, match="leakage"):
        assert_no_group_overlap([values[0]], [values[1]])


def test_block_split_purges_neighbor_frames():
    values = records(10)
    train_idx, val_idx = next(make_folds(values, n_splits=2, mode="block", purge_frames=1))
    assert int(val_idx[-1]) + 1 not in set(map(int, train_idx))

