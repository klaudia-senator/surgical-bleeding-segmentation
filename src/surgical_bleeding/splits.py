from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, KFold


@dataclass(frozen=True)
class PairRecord:
    image_path: Path
    mask_path: Path
    group: str


def source_group(path: Path, root: Path, strategy: str) -> str:
    relative = path.relative_to(root)
    if strategy == "parent":
        return relative.parts[0] if len(relative.parts) > 1 else path.parent.name
    if strategy == "grandparent":
        return relative.parts[0] if len(relative.parts) > 2 else path.parent.name
    if strategy == "stem_prefix":
        return path.stem.split("_")[0]
    raise ValueError(f"Unsupported grouping strategy: {strategy}")


def make_folds(records: list[PairRecord], n_splits: int, mode: str, seed: int = 42, purge_frames: int = 0):
    indices = np.arange(len(records))
    if mode == "random":
        yield from KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(indices)
        return
    if mode == "group":
        groups = np.asarray([record.group for record in records])
        if len(np.unique(groups)) < n_splits:
            raise ValueError("Fewer source groups than folds")
        yield from GroupKFold(n_splits=n_splits).split(indices, groups=groups)
        return
    if mode == "block":
        blocks = np.array_split(indices, n_splits)
        for validation in blocks:
            blocked = set(int(i) for i in validation)
            if purge_frames:
                lo, hi = int(validation[0]), int(validation[-1])
                blocked.update(range(max(0, lo - purge_frames), min(len(records), hi + purge_frames + 1)))
            train = np.asarray([i for i in indices if int(i) not in blocked], dtype=int)
            yield train, validation
        return
    raise ValueError(f"Unknown split mode: {mode}")


def assert_no_group_overlap(train: list[PairRecord], validation: list[PairRecord]) -> None:
    overlap = {record.group for record in train} & {record.group for record in validation}
    if overlap:
        raise ValueError(f"Source leakage detected; groups occur in train and validation: {sorted(overlap)}")

