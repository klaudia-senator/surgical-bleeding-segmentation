from __future__ import annotations

import torch
from torch import nn


class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = float(dice_weight)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target_hw: torch.Tensor) -> torch.Tensor:
        target = target_hw.unsqueeze(1).float()
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * target).sum(dim=(2, 3))
        denominator = probabilities.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + self.eps) / (denominator + self.eps)).mean()
        return self.bce(logits, target) + self.dice_weight * dice_loss

