from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from surgical_bleeding.data import IMAGE_EXTENSIONS, collect_files


def binarize_mask(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    foreground = np.any(array[..., :3] > 0, axis=2) if array.ndim == 3 else array > 0
    return (foreground.astype(np.uint8) * 255)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CVAT raster masks to binary PNG masks.")
    parser.add_argument("--input-masks", type=Path, required=True)
    parser.add_argument("--output-masks", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, help="Optional image directory used only for pairing warnings.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    masks = collect_files(args.input_masks, args.recursive)
    if not masks:
        raise FileNotFoundError(f"No masks found in {args.input_masks}")
    image_stems = {p.stem for p in collect_files(args.images_dir, args.recursive)} if args.images_dir else set()
    args.output_masks.mkdir(parents=True, exist_ok=True)
    missing = converted = skipped = 0
    for path in masks:
        if image_stems and path.stem not in image_stems:
            print(f"[WARN] no matching image: {path.name}")
            missing += 1
        output = args.output_masks / f"{path.stem}.png"
        if output.exists() and not args.overwrite:
            skipped += 1
            continue
        Image.fromarray(binarize_mask(path)).save(output)
        converted += 1
    print(f"Converted={converted} skipped={skipped} pairing_warnings={missing}")


if __name__ == "__main__":
    main()

