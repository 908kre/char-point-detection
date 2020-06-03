import numpy as np
import typing as t
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from itertools import product as product
import torchvision

from .bifpn import BiFPN

#  from .efficientnet import EfficientNet
from .retinahead import RetinaHead
from .focalloss import FocalLoss

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
        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)


class Anchors(nn.Module):
    def __init__(
        self,
        pyramid_levels: t.List[int] = [3, 4, 5, 6, 7],
        ratios: t.List[float] = [0.5, 1, 2],
        scales: t.List[float] = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
        strides: t.Optional[t.List[int]] = None,
        sizes: t.Optional[t.List[int]] = None,
    ) -> None:
        super(Anchors, self).__init__()

        self.pyramid_levels = pyramid_levels
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        else:
            self.strides = strides
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        else:
            self.sizes = sizes
        self.ratios = ratios
        self.scales = scales

    def forward(self, image):  # type:ignore

        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [
            (image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels
        ]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(
                base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales
            )
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        return torch.from_numpy(all_anchors.astype(np.float32)).to(image.device)


def shift(shape: t.Any, stride: int, anchors: t.Any) -> t.Any:
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack(
        (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())
    ).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose(
        (1, 0, 2)
    )
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(
    base_size: int = 16,
    ratios: t.List[float] = [0.5, 1, 2],
    scales: t.List[float] = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
) -> t.Any:
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


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
        ...
        super().__init__()


#          self.backbone = EfficientNet.from_pretrained(network)
#  self.neck = BiFPN(
#      in_channels=self.backbone.get_list_features()[-5:],
#      out_channels=W_bifpn,
#      stack=D_bifpn,
#      num_outs=5,
#  )
#          self.bbox_head = RetinaHead(num_classes=num_classes, in_channels=W_bifpn)
#
#          self.anchors = Anchors()
#          self.regressBoxes = BBoxTransform()
#          self.clipBoxes = ClipBoxes()
#          self.threshold = threshold
#          self.iou_threshold = iou_threshold
#          for m in self.modules():
#              if isinstance(m, nn.Conv2d):
#                  n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                  m.weight.data.normal_(0, math.sqrt(2.0 / n))
#              elif isinstance(m, nn.BatchNorm2d):
#                  m.weight.data.fill_(1)
#                  m.bias.data.zero_()
#          self.freeze_bn()
#          self.criterion = FocalLoss()
#
#      def forward(self, inputs):  # type: ignore
#          if self.is_training:
#              inputs, annotations = inputs
#          else:
#              inputs = inputs
#          x = self.extract_feat(inputs)
#          outs = self.bbox_head(x)
#          classification = torch.cat([out for out in outs[0]], dim=1)
#          regression = torch.cat([out for out in outs[1]], dim=1)
#          anchors = self.anchors(inputs)
#          if self.is_training:
#              return self.criterion(classification, regression, anchors, annotations)
#          else:
#              transformed_anchors = self.regressBoxes(anchors, regression)
#              transformed_anchors = self.clipBoxes(transformed_anchors, inputs)
#              scores = torch.max(classification, dim=2, keepdim=True)[0]
#              scores_over_thresh = (scores > self.threshold)[0, :, 0]
#
#              if scores_over_thresh.sum() == 0:
#                  print("No boxes to NMS")
#                  # no boxes to NMS, just return
#                  return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
#              classification = classification[:, scores_over_thresh, :]
#              transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
#              scores = scores[:, scores_over_thresh, :]
#              anchors_nms_idx = nms(
#                  transformed_anchors[0, :, :],
#                  scores[0, :, 0],
#                  iou_threshold=self.iou_threshold,
#              )
#              nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
#              return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]
#
#      def freeze_bn(self) -> None:
#          """Freeze BatchNorm layers."""
#          for layer in self.modules():
#              if isinstance(layer, nn.BatchNorm2d):
#                  layer.eval()
#
#      def extract_feat(self, img: Tensor) -> Tensor:
#          """
#              Directly extract features from the backbone+neck
#          """
#          x = self.backbone(img)
#          x = self.neck(x[-5:])
#          return x
