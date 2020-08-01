import numpy as np
import torch
from torchvision.ops.boxes import box_iou
from sklearn.metrics import average_precision_score, precision_recall_curve, auc


def coco_to_pascal(boxes):
    boxes_ = boxes.clone()
    boxes_[..., 2:] += boxes_[..., :2]
    return boxes_


def sweep_average_precision(
    true_boxes,
    pred_boxes,
    confidences,
    iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
):
    """
    compute ap, precision @ f1-best, recall @ f1-best.
    boxes are expected to be in pascal voc (x1, y1, x2, y2) format.
    """
    pred_boxes = pred_boxes[torch.argsort(confidences, descending=True)]
    if len(true_boxes) == 0:
        if len(pred_boxes) == 0:
            ap = 1.0
            precision = 1.0
            recall = 1.0
            threshold = 0.5
        else:
            ap = 0.0
            precision = 0.0
            recall = 0.0
            threshold = 0.5
        return ap, precision, recall, threshold
    else:
        if len(pred_boxes) == 0:
            return 0.0, 0.0, 0.0, 0.5

    iou_matrix = box_iou(pred_boxes, true_boxes)

    ap, precision, recall, threshold = [], [], [], []
    for iou_threshold in iou_thresholds:
        ap_, precision_, recall_, threshold_ = compute_average_precision(
            iou_matrix, iou_threshold, confidences
        )
        ap.append(ap_)
        precision.append(precision_)
        recall.append(recall_)
        threshold.append(threshold_)

    return np.mean(ap), np.mean(precision), np.mean(recall), np.mean(threshold)


def compute_average_precision(
    iou_matrix: torch.Tensor, iou_threshold: float, confidences: torch.Tensor
) -> float:
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
    y_true = gt[: n_pred + fn]
    probas_pred = scores[: n_pred + fn]
    precision, recall, thresholds = precision_recall_curve(y_true, probas_pred)
    ap = auc(recall, precision)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    best_idx = np.argmax(f1_score)
    rep_precision, rep_recall = precision[best_idx], recall[best_idx]
    rep_threshold = thresholds[best_idx]
    return ap, rep_precision, rep_recall, rep_threshold
