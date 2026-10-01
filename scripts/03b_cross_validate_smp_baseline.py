from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from surgical_bleeding.data import BloodSegmentationDataset, read_pairs_csv
from surgical_bleeding.losses import BCEDiceLoss
from surgical_bleeding.metrics import summarize_folds
from surgical_bleeding.models import make_smp_model
from surgical_bleeding.reproducibility import save_run_metadata, set_seed
from surgical_bleeding.training import evaluate, train_epoch, write_rows


def freeze_batchnorm(module: torch.nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            layer.eval()


def train_fold(fold_dir: Path, output: Path, args, device: str) -> dict:
    fold_number = int(fold_dir.name.rsplit("_", 1)[-1])
    set_seed(args.seed + fold_number, args.deterministic)
    train_ds = BloodSegmentationDataset(read_pairs_csv(fold_dir / "train.csv"), args.img_size, augment=True)
    val_ds = BloodSegmentationDataset(read_pairs_csv(fold_dir / "val.csv"), args.img_size, augment=False)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device == "cuda")
    train_eval_loader = DataLoader(train_ds, 1, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, 1, shuffle=False, num_workers=args.num_workers)
    model = make_smp_model(args.model, args.encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = BCEDiceLoss(dice_weight=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")
    fold_output = output / fold_dir.name
    fold_output.mkdir(parents=True, exist_ok=True)
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        if args.model == "deeplabv3plus" and args.batch_size == 1:
            freeze_batchnorm(model)
        train_epoch(model, train_loader, optimizer, scaler, device, args.amp, criterion)
        train_metrics = evaluate(model, train_eval_loader, device, args.threshold, criterion)
        val_metrics = evaluate(model, val_loader, device, args.threshold, criterion)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}, "seconds": time.time() - started}
        history.append(row)
        if best is None or row["val_dice"] > best["val_dice"]:
            best = row.copy(); torch.save(model.state_dict(), fold_output / "best_model.pt")
        torch.save(model.state_dict(), fold_output / "last_model.pt")
        print(f"fold={fold_number} epoch={epoch} val_dice={row['val_dice']:.4f} val_iou={row['val_iou']:.4f}")
    write_rows(fold_output / "history.csv", history)
    assert best is not None
    metrics = {"fold": fold_number, "model": args.model, "encoder": args.encoder, "n_train": len(train_ds), "n_val": len(val_ds), "best_epoch": best["epoch"], "best_val_loss": best["val_loss"], "best_val_dice": best["val_dice"], "best_val_iou": best["val_iou"]}
    (fold_output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validate public SMP segmentation baselines.")
    parser.add_argument("--model", choices=["unetplusplus", "deeplabv3plus"], required=True)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=512); parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0); parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="Prefer deterministic kernels; may be slower.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(args.out_dir / "run_metadata.json", vars(args))
    folds = sorted((p for p in args.folds_dir.glob("fold_*") if p.is_dir()), key=lambda p: int(p.name.rsplit("_", 1)[-1]))
    if not folds:
        raise FileNotFoundError(f"No fold directories in {args.folds_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = [train_fold(fold, args.out_dir, args, device) for fold in folds]
    frame = pd.DataFrame(records); frame.to_csv(args.out_dir / "fold_metrics.csv", index=False)
    dice_mean, dice_std = summarize_folds(frame["best_val_dice"].to_numpy())
    iou_mean, iou_std = summarize_folds(frame["best_val_iou"].to_numpy())
    loss_mean, loss_std = summarize_folds(frame["best_val_loss"].to_numpy())
    summary = {"model": args.model, "encoder": args.encoder, "n_folds": len(frame), "std_definition": "sample SD (ddof=1)", "dice_mean": dice_mean, "dice_std": dice_std, "iou_mean": iou_mean, "iou_std": iou_std, "val_loss_mean": loss_mean, "val_loss_std": loss_std}
    (args.out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(args.out_dir / "cv_summary.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

