import torch
import numpy as np
import typing as t
import math
import torch.nn.functional as F
from app import config
from torch import nn, Tensor
from logging import getLogger
from .modules import ConvBR2d
from .bottlenecks import SENextBottleneck2d
from .bifpn import BiFPN, FP
from .backbones import EfficientNetBackbone, ResNetBackbone
from app.entities import ImageBatch
from torchvision.ops import nms

from pathlib import Path
import albumentations as albm

logger = getLogger(__name__)


class Reg(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()
        channels = in_channels
        self.conv = nn.Sequential(
            *[SENextBottleneck2d(in_channels, in_channels) for _ in range(depth)]
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0), nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = self.conv(x)
        x = self.out(x)
        return x


Heatmap = t.NewType("Heatmap", Tensor)
Sizemap = t.NewType("Sizemap", Tensor)
NetOutput = t.Tuple[Heatmap, Sizemap]


class CenterNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = 32
        self.backbone = EfficientNetBackbone(1, out_channels=channels)
        self.fpn = nn.Sequential(BiFPN(channels=channels))
        self.heatmap = Reg(in_channels=channels, out_channels=1, depth=2)
        self.box_size = Reg(in_channels=channels, out_channels=2, depth=2)

    def forward(self, x: ImageBatch) -> NetOutput:
        fp = self.backbone(x)
        fp = self.fpn(fp)
        heatmap = self.heatmap(fp[0])
        sizemap = self.box_size(fp[0])
        return Heatmap(heatmap), Sizemap(sizemap)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        alpha = self.alpha
        beta = self.beta
        pred = torch.clamp(pred, min=self.eps, max=1 - self.eps)
        pos_mask = gt.eq(1).float()
        neg_mask = gt.lt(1).float()
        pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
        neg_loss = (
            -((1 - gt) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg_mask
        )
        loss = (pos_loss + neg_loss).sum()
        num_pos = pos_mask.sum().float()
        return loss / num_pos

class Criterion:
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.focal_loss = FocalLoss()
        self.reg_loss = RegLoss()

    def __call__(self, src: NetOutput, tgt: NetOutput) -> Tensor:
        s_hm, s_sm = src
        t_hm, t_sm = tgt
        hm_loss = self.focal_loss(s_hm, t_hm)
        size_loss = self.reg_loss(s_sm, t_sm) * 10
        return hm_loss + size_loss

class RegLoss:
    def __call__(self, output: Sizemap, target: Sizemap,) -> Tensor:
        mask = (target > 0).view(target.shape)
        num = mask.sum()
        regr_loss = F.l1_loss(output, target, reduction="none") * mask
        regr_loss = regr_loss.sum() / (num + 1e-4)
        return regr_loss

