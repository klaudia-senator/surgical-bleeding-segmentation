from __future__ import annotations

import numpy as np
import torch


def dice_iou_binary(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    pred_b, target_b = pred.astype(bool), target.astype(bool)
    if not pred_b.any() and not target_b.any():
        return 1.0, 1.0
    intersection = np.logical_and(pred_b, target_b).sum()
    union = np.logical_or(pred_b, target_b).sum()
    dice = (2.0 * intersection + eps) / (pred_b.sum() + target_b.sum() + eps)
    return float(dice), float((intersection + eps) / (union + eps))


@torch.no_grad()
def batch_dice_iou_from_logits(logits: torch.Tensor, target_hw: torch.Tensor, threshold: float = 0.5) -> tuple[float, float]:
    pred = (torch.sigmoid(logits)[:, 0] >= threshold).cpu().numpy()
    target = (target_hw >= 0.5).cpu().numpy()
    values = [dice_iou_binary(p, y) for p, y in zip(pred, target)]
    return float(np.mean([v[0] for v in values])), float(np.mean([v[1] for v in values]))


def summarize_folds(values: np.ndarray) -> tuple[float, float]:
    """Return mean and sample SD (ddof=1), or zero SD for a single fold."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0

