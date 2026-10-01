from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .losses import BCEDiceLoss
from .metrics import batch_dice_iou_from_logits


@torch.no_grad()
def evaluate(model, dataloader, device: str, threshold: float, criterion: BCEDiceLoss) -> dict[str, float]:
    model.eval()
    losses, dices, ious = [], [], []
    for batch in dataloader:
        images, targets = batch["image"].to(device), batch["mask"].to(device)
        logits = model(images)
        losses.append(float(criterion(logits, targets)))
        dice, iou = batch_dice_iou_from_logits(logits, targets, threshold)
        dices.append(dice)
        ious.append(iou)
    return {"loss": float(np.mean(losses)), "dice": float(np.mean(dices)), "iou": float(np.mean(ious))}


def train_epoch(model, dataloader, optimizer, scaler, device: str, amp: bool, criterion: BCEDiceLoss) -> float:
    model.train()
    losses = []
    for batch in dataloader:
        images, targets = batch["image"].to(device), batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device == "cuda"):
            loss = criterion(model(images), targets)
        if not torch.isfinite(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
    if not losses:
        raise RuntimeError("No finite training loss was produced")
    return float(np.mean(losses))


def write_rows(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

