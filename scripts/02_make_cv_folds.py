from __future__ import annotations

import argparse
import csv
from pathlib import Path

from surgical_bleeding.data import pair_by_stem
from surgical_bleeding.splits import PairRecord, assert_no_group_overlap, make_folds, source_group


def write_csv(path: Path, records: list[PairRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "mask_path", "group"])
        writer.writeheader()
        writer.writerows({"image_path": str(r.image_path.resolve()), "mask_path": str(r.mask_path.resolve()), "group": r.group} for r in records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-aware segmentation CV manifests.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-suffix", default="")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--split-mode", choices=["random", "group", "block"], default="group")
    parser.add_argument("--group-from", choices=["parent", "grandparent", "stem_prefix"], default="parent")
    parser.add_argument("--purge-frames", type=int, default=0, help="For block CV, remove this many neighboring frames from training at validation boundaries.")
    args = parser.parse_args()
    pairs = pair_by_stem(args.images_dir, args.masks_dir, args.mask_suffix, args.recursive)
    if len(pairs) < args.n_splits:
        raise ValueError("Fewer matched pairs than folds")
    records = [PairRecord(image, mask, source_group(image, args.images_dir, args.group_from) if args.split_mode == "group" else "none") for image, mask in pairs]
    output = args.out_dir / f"{args.dataset_name}_{args.n_splits}fold"
    write_csv(output / "all_pairs.csv", records)
    summary = []
    for number, (train_idx, val_idx) in enumerate(make_folds(records, args.n_splits, args.split_mode, args.seed, args.purge_frames), 1):
        train, validation = [records[int(i)] for i in train_idx], [records[int(i)] for i in val_idx]
        if args.split_mode == "group":
            assert_no_group_overlap(train, validation)
        write_csv(output / f"fold_{number}" / "train.csv", train)
        write_csv(output / f"fold_{number}" / "val.csv", validation)
        summary.append({"fold": number, "n_train": len(train), "n_val": len(validation), "train_groups": ";".join(sorted({r.group for r in train})), "val_groups": ";".join(sorted({r.group for r in validation}))})
    with (output / "fold_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    print(f"Saved {len(summary)} folds for {len(records)} pairs to {output}")


if __name__ == "__main__":
    main()

