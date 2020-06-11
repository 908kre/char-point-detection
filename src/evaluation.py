import numpy as np
import torch
from torchvision.ops.boxes import box_iou
from sklearn.metrics import average_precision_score


def coco_to_pascal(boxes):
    boxes_ = boxes.clone()
    boxes_[..., 2:] += boxes_[..., :2]
    return boxes_


def sweep_average_precision(
        true_boxes, pred_boxes, confidences,
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    boxes are expected to be in pascal voc (x1, y1, x2, y2) format.
    """
    pred_boxes = pred_boxes[torch.argsort(confidences, descending=True)]
    if len(true_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0
    if len(pred_boxes) == 0:
        return 0.0
    iou_matrix = box_iou(pred_boxes, true_boxes)
    return np.mean([
        compute_average_precision(iou_matrix, iou_threshold, confidences)
        for iou_threshold in iou_thresholds
    ])


def compute_average_precision(
        iou_matrix: torch.Tensor, iou_threshold: float, confidences: torch.Tensor) -> float:
    iou_matrix = iou_matrix.clone()
    n_pred = iou_matrix.size(0)
    n_true = iou_matrix.size(1)
    gt = np.ones(n_pred + n_true)
    scores = np.zeros(n_pred + n_true)
    match = 0
    for pred_idx in range(n_pred):
        scores[pred_idx] = confidences[pred_idx]
        max_iou, max_idx = iou_matrix[pred_idx].max(0)
        if max_iou > iou_threshold:
            gt[pred_idx] = 1
            match += 1
            iou_matrix[:, max_idx] = 0
        else:
            gt[pred_idx] = 0
    fn = n_true - match
    return average_precision_score(gt[:n_pred + fn], scores[:n_pred + fn])
