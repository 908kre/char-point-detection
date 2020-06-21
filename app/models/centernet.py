import torch
import numpy as np
import typing as t
import math
import torch.nn.functional as F
from app import config
from torch import nn, Tensor
from logging import getLogger
from app.entities import YoloBoxes, Confidences
from .modules import ConvBR2d
from .bottlenecks import SENextBottleneck2d
from .bifpn import BiFPN, FP
from .backbones import EfficientNetBackbone, ResNetBackbone
from app.entities import ImageBatch, PredBoxes, Image, Batch
from torchvision.ops import nms

from pathlib import Path
import albumentations as albm

logger = getLogger(__name__)


def collate_fn(batch: Batch) -> t.Tuple[ImageBatch, t.List[YoloBoxes], t.List[str]]:
    images: t.List[t.Any] = []
    id_batch: t.List[str] = []
    box_batch: t.List[YoloBoxes] = []

    for id, img, boxes in batch:
        images.append(img)
        box_batch.append(boxes)
        id_batch.append(id)
    return ImageBatch(torch.stack(images)), box_batch, id_batch


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
    def __init__(self,) -> None:
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


def gaussian_2d(shape: t.Any, sigma: float = 1) -> np.ndarray:
    m, n = int((shape[0] - 1.0) / 2.0), int((shape[1] - 1.0) / 2.0)
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


class ToBoxes:
    def __init__(self, thresold: float, limit: int = 100) -> None:
        self.limit = limit
        self.thresold = thresold

    def __call__(self, inputs: NetOutput) -> t.List[t.Tuple[YoloBoxes, Confidences]]:
        heatmaps, sizemaps = inputs
        device = heatmaps.device
        kp_maps = (F.max_pool2d(heatmaps, 3, stride=1, padding=1) == heatmaps) & (
            heatmaps > self.thresold
        )
        batch_size, _, height, width = heatmaps.shape
        original_wh = torch.tensor([width, height], dtype=torch.float32).to(device)
        rows: t.List[t.Tuple[YoloBoxes, Confidences]] = []
        for hm, kp_map, size_map in zip(
            heatmaps.squeeze(1), kp_maps.squeeze(1), sizemaps
        ):
            pos = kp_map.nonzero()
            confidences = hm[pos[:, 0], pos[:, 1]]
            wh = size_map[:, pos[:, 0], pos[:, 1]]
            cxcy = pos[:, [1, 0]].float() / original_wh
            boxes = torch.cat([cxcy, wh.permute(1, 0)], dim=1)
            sort_idx = confidences.argsort(descending=True)[: self.limit]
            rows.append(
                (YoloBoxes(boxes[sort_idx]), Confidences(confidences[sort_idx]))
            )
        return rows


class SoftHeatMap:
    def __init__(
        self, mount_size: t.Tuple[int, int] = (5, 5), sigma: float = 1,
    ) -> None:
        self.mount_size = mount_size
        self.mount_pad = (
            self.mount_size[0] % 2,
            self.mount_size[1] % 2,
        )
        mount = gaussian_2d(self.mount_size, sigma=sigma)
        self.mount = torch.tensor(mount, dtype=torch.float32).view(
            1, 1, mount.shape[0], mount.shape[1]
        )
        self.mount = self.mount / self.mount.max()

    def __call__(self, boxes: YoloBoxes, size: t.Tuple[int, int]) -> NetOutput:
        device = boxes.device
        w, h = size
        heatmap = torch.zeros((1, 1, h, w), dtype=torch.float32).to(device)
        sizemap = torch.zeros((1, 2, h, w), dtype=torch.float32).to(device)
        box_count, _ = boxes.shape
        if box_count == 0:
            return Heatmap(heatmap), Sizemap(sizemap)
        box_cx, box_cy, box_w, box_h = torch.unbind(boxes, dim=1)
        box_cx = (box_cx * w).long()
        box_cy = (box_cy * h).long()
        sizemap[:, :, box_cy, box_cx] = torch.stack([box_w, box_h], dim=0)
        mount_w, mount_h = self.mount_size
        pad_w, pad_h = self.mount_pad
        mount_x0 = box_cx - mount_h // 2
        mount_x1 = box_cx + mount_h // 2 + pad_h
        mount_y0 = box_cy - mount_w // 2
        mount_y1 = box_cy + mount_w // 2 + pad_w

        mount = self.mount.to(device)
        for x0, x1, y0, y1 in zip(mount_x0, mount_x1, mount_y0, mount_y1):
            target = heatmap[:, :, y0:y1, x0:x1]  # type: ignore
            _, _, target_h, target_w = target.shape
            if (target_h >= mount_h) and (target_w >= mount_w):
                mount = torch.max(mount, target)
                heatmap[:, :, y0:y1, x0:x1] = mount  # type: ignore
        return Heatmap(heatmap), Sizemap(sizemap)


class PreProcess:
    def __init__(self, device: t.Any) -> None:
        super().__init__()
        self.heatmap = SoftHeatMap()
        self.device = device

    def __call__(
        self, batch: t.Tuple[ImageBatch, t.List[YoloBoxes]]
    ) -> t.Tuple[ImageBatch, NetOutput]:
        image_batch, boxes_batch = batch
        image_batch = ImageBatch(image_batch.to(self.device))
        hms: t.List[t.Any] = []
        sms: t.List[t.Any] = []
        _, _, h, w = image_batch.shape
        for img, boxes in zip(image_batch.unbind(0), boxes_batch):
            hm, sm = self.heatmap(YoloBoxes(boxes.to(self.device)), (w // 2, h // 2))
            hms.append(hm)
            sms.append(sm)

        heatmap = torch.cat(hms, dim=0)
        sizemap = torch.cat(sms, dim=0)
        return image_batch, (Heatmap(heatmap), Sizemap(sizemap))
