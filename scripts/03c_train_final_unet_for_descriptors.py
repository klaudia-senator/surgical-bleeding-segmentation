from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from surgical_bleeding.data import BloodSegmentationDataset, pair_by_stem, read_pairs_csv
from surgical_bleeding.losses import BCEDiceLoss
from surgical_bleeding.models import load_owned_unet_class
from surgical_bleeding.reproducibility import save_run_metadata, set_seed
from surgical_bleeding.training import evaluate, train_epoch, write_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the owned binary U-Net on all selected data for downstream descriptors.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pairs-csv", type=Path)
    source.add_argument("--images-dir", type=Path)
    parser.add_argument("--masks-dir", type=Path); parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-import", default="src.models.unet:UNetBinary")
    parser.add_argument("--img-size", type=int, default=512); parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1); parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true"); parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if args.images_dir and not args.masks_dir:
        parser.error("--masks-dir is required with --images-dir")
    pairs = read_pairs_csv(args.pairs_csv) if args.pairs_csv else pair_by_stem(args.images_dir, args.masks_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed, args.deterministic); save_run_metadata(args.out_dir / "run_metadata.json", vars(args))
    model_class = load_owned_unet_class(args.model_import)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BloodSegmentationDataset(pairs, args.img_size, augment=True)
    loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    descriptive_loader = DataLoader(BloodSegmentationDataset(pairs, args.img_size, augment=False), 1, shuffle=False, num_workers=args.num_workers)
    model = model_class(base_ch=64, dropout=args.dropout, use_batchnorm=True, upsample_mode="transpose").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = BCEDiceLoss(dice_weight=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time(); step_loss = train_epoch(model, loader, optimizer, scaler, device, args.amp, criterion)
        descriptive = evaluate(model, descriptive_loader, device, args.threshold, criterion)
        history.append({"epoch": epoch, "train_step_loss": step_loss, "descriptive_train_loss": descriptive["loss"], "descriptive_train_dice": descriptive["dice"], "descriptive_train_iou": descriptive["iou"], "seconds": time.time() - started})
        torch.save(model.state_dict(), args.out_dir / "last_model.pt")
    write_rows(args.out_dir / "history.csv", history); torch.save(model.state_dict(), args.out_dir / "final_model.pt")
    summary = {"purpose": "all-data fit for downstream temporal descriptor extraction", "performance_note": "These are descriptive training-set fit metrics, not an independent generalization estimate. Use cross-validation results for performance estimation.", "n_pairs": len(pairs), "epochs": args.epochs, "descriptive_train_dice": history[-1]["descriptive_train_dice"], "descriptive_train_iou": history[-1]["descriptive_train_iou"]}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

