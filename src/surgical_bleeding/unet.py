from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 convolutions. Optional BatchNorm and Dropout2d are configurable."""

    def __init__(self, in_ch: int, out_ch: int, use_batchnorm: bool = True, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_batchnorm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_ch, out_ch, use_batchnorm=use_batchnorm, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_batchnorm: bool = True, dropout: float = 0.0, upsample_mode: str = "transpose"):
        super().__init__()
        if upsample_mode == "transpose":
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            conv_in = in_ch // 2 + skip_ch
        elif upsample_mode == "bilinear":
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            conv_in = in_ch + skip_ch
        else:
            raise ValueError("upsample_mode must be 'transpose' or 'bilinear'")
        self.conv = DoubleConv(conv_in, out_ch, use_batchnorm=use_batchnorm, dropout=dropout)

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x_up)
        diff_y = x_skip.size(2) - x_up.size(2)
        diff_x = x_skip.size(3) - x_up.size(3)
        if diff_y != 0 or diff_x != 0:
            x_up = nn.functional.pad(x_up, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x_skip, x_up], dim=1))


class UNetBinary(nn.Module):
    """In-house U-Net used for binary blood segmentation in the associated paper."""

    def __init__(self, base_ch: int = 64, dropout: float = 0.3, use_batchnorm: bool = True, upsample_mode: str = "transpose"):
        super().__init__()
        drop_shallow, drop_deep = 0.0, float(dropout)
        self.inc = DoubleConv(3, base_ch, use_batchnorm=use_batchnorm, dropout=drop_shallow)
        self.down1 = Down(base_ch, base_ch * 2, use_batchnorm=use_batchnorm, dropout=drop_shallow)
        self.down2 = Down(base_ch * 2, base_ch * 4, use_batchnorm=use_batchnorm, dropout=drop_deep)
        self.down3 = Down(base_ch * 4, base_ch * 8, use_batchnorm=use_batchnorm, dropout=drop_deep)
        bottleneck_ch = base_ch * 16
        self.bot = DoubleConv(base_ch * 8, bottleneck_ch, use_batchnorm=use_batchnorm, dropout=drop_deep)
        self.up3 = Up(bottleneck_ch, base_ch * 8, base_ch * 8, use_batchnorm=use_batchnorm, dropout=drop_deep, upsample_mode=upsample_mode)
        self.up2 = Up(base_ch * 8, base_ch * 4, base_ch * 4, use_batchnorm=use_batchnorm, dropout=drop_deep, upsample_mode=upsample_mode)
        self.up1 = Up(base_ch * 4, base_ch * 2, base_ch * 2, use_batchnorm=use_batchnorm, dropout=drop_shallow, upsample_mode=upsample_mode)
        self.up0 = Up(base_ch * 2, base_ch, base_ch, use_batchnorm=use_batchnorm, dropout=drop_shallow, upsample_mode=upsample_mode)
        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        xb = self.bot(x3)
        d3 = self.up3(xb, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)
        return self.head(self.up0(d1, x0))
