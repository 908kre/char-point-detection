#  import typing as t
#  import torch
#  import numpy as np
#  from collections import defaultdict
#  from torch import Tensor
#  from torchvision.ops.boxes import box_iou
#
#
#  def precition(iou_matrix: Tensor, threshold: float) -> float:
#      candidates, candidate_ids = (iou_matrix).max(1)
#      n_pr, n_gt = iou_matrix.shape
#      match_ids = candidate_ids[candidates > threshold]
#      fp = n_pr - len(match_ids)
#      (tp,) = torch.unique(match_ids).shape  # type: ignore
#      fn = n_gt - tp
#      return tp / (fp + tp + fn)
#
#
#  class MeamPrecition:
#      def __init__(
#          self, iou_thresholds: t.List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
#      ) -> None:
#          self.iou_thresholds = iou_thresholds
#
#      def __call__(self, pred_boxes: Tensor, gt_boxes: Tensor) -> float:
#          if len(gt_boxes) == 0:
#              return 1.0 if len(pred_boxes) == 0 else 0.0
#          if len(pred_boxes) == 0:
#              return 0.0
#          iou_matrix = box_iou(pred_boxes, gt_boxes)
#          res = np.mean([precition(iou_matrix, t) for t in self.iou_thresholds])
#          return res
#
#
#  class Evaluate:
#      def __init__(self) -> None:
#          self.mean_precision = MeamPrecition()
#
#      def __call__(self, pred: Annotations, gt: Annotations) -> float:
#          lenght = len(gt)
#          score = 0.0
#          for pred_boxes, gt_boxes in zip(pred, gt):
#              device = pred_boxes.device
#              gt_boxes = gt_boxes.to(device)
#              pred_boxes = pred_boxes.to_xyxy()
#              gt_boxes = gt_boxes.to_xyxy()
#              score += self.mean_precision(pred_boxes.boxes, gt_boxes.boxes)
#          return score / lenght
