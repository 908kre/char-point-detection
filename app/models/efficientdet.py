import numpy as np
import typing as t
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import nms
import math
from itertools import product as product
import torchvision
from logging import getLogger
from .bifpn import BiFPN

#  from .efficientnet import EfficientNet
from .losses import FocalLoss
from .anchors import Anchors

logger = getLogger(__name__)

ModelName = t.Literal[
    "efficientdet-d0",
    "efficientdet-d1",
    "efficientdet-d2",
    "efficientdet-d3",
    "efficientdet-d4",
    "efficientdet-d5",
    "efficientdet-d6",
    "efficientdet-d7",
]


class BBoxTransform(nn.Module):
    def __init__(
        self, mean: t.Optional[t.Any] = None, std: t.Optional[t.Any] = None
    ) -> None:
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(
                np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)
            )
        else:
            self.std = std

    def forward(self, boxes, deltas):  # type: ignore

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]
        #  print(dx, dy,dw,dh)

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack(
            [pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2
        )

        return pred_boxes


class ClipBoxes(nn.Module):
    def __init__(
        self, width: t.Optional[int] = None, height: t.Optional[int] = None
    ) -> None:
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):  # type: ignore
        """
        boxes: [B, C, 4] [xmin, ymin, xmax, ymax]
        img: [B, C, W, H]
        """

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


class ClassificationModel(nn.Module):
    def __init__(
        self,
        num_features_in: int,
        num_anchors: int = 9,
        num_classes: int = 80,
        prior: float = 0.01,
        feature_size: int = 256,
    ) -> None:
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(
            feature_size, num_anchors * num_classes, kernel_size=3, padding=1
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x):  # type: ignore
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        out = self.output_act(out)
        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class RegressionModel(nn.Module):
    def __init__(
        self, num_features_in: int, num_anchors: int = 9, feature_size: int = 256
    ) -> None:
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)


class EfficientDet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        network: ModelName = "efficientdet-d0",
        D_bifpn: int = 3,
        W_bifpn: int = 88,
        D_class: int = 3,
        threshold: float = 0.01,
        iou_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.anchors = Anchors()
        self.clip_boxes = ClipBoxes()
        self.neck = BiFPN(channels=128)
        self.threshold = threshold
        self.regression = RegressionModel(128)
        self.classification = ClassificationModel(128, num_classes=2)
        self.bbox_transform = BBoxTransform()
        self.iou_threshold = iou_threshold
        self.criterion = FocalLoss()

    def forward(
        self, inputs: Tensor, annotations: t.Optional[Tensor] = None
    ) -> t.Tuple[Tensor, Tensor, Tensor]:
        features = self.extract_feat(inputs)
        regressions = torch.cat(
            [self.regression(feature) for feature in features], dim=1
        )
        classifications = torch.cat(
            [self.classification(feature) for feature in features], dim=1
        )
        anchors = self.anchors(inputs)
        if annotations is not None:
            return self.criterion(classifications, regressions, anchors, annotations)

        if annotations is None:
            transformed_anchors = self.bbox_transform(anchors, regressions)
            transformed_anchors = self.clip_boxes(transformed_anchors, inputs)
            scores = torch.max(classifications, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > self.threshold)[0, :, 0]
            if scores_over_thresh.sum() == 0:
                logger.info("No boxes to NMS")
                return torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)
            classification = classifications[:, scores_over_thresh, :]
            anchors_nms_idx = nms(
                transformed_anchors[0, :, :],
                scores[0, :, 0],
                iou_threshold=self.iou_threshold,
            )
            nms_scores, nms_class = classifications[0, anchors_nms_idx, :].max(dim=1)
            return nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]

    def extract_feat(self, x: Tensor) -> t.List[Tensor]:
        features = [
            torch.empty(1, 128, 1024 // (2 ** i), 1024 // (2 ** i)) for i in range(8)
        ][-5:]
        return self.neck(features)
