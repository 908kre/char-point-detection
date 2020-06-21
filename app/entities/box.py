import torch
import typing as t
from torch import Tensor

CoCoBoxes = t.NewType(
    "CoCoBoxes", Tensor
)  # [B, Pos] Pos:[x0, y0, width, height] original
YoloBoxes = t.NewType(
    "YoloBoxes", Tensor
)  # [B, Pos] Pos:[cx, cy, width, height] normalized
PascalBoxes = t.NewType("PascalBoxes", Tensor)  # [B, Pos] Pos:[x0, y0, x1, y1] original

Labels = t.NewType("Labels", Tensor)
Confidences = t.NewType("Confidences", Tensor)

PredBoxes = t.Tuple[CoCoBoxes, Confidences]
LabelBoxes = t.Tuple[CoCoBoxes, Labels]
Size = t.Tuple[int, int]


def yoyo_to_pascal(x: CoCoBoxes, size: t.Tuple[int, int]) -> YoloBoxes:
    ...


def coco_to_yolo(coco: CoCoBoxes, size: t.Tuple[int, int]) -> YoloBoxes:
    size_w, size_h = size
    x0, y0, x1, y1 = coco_to_pascal(coco).unbind(-1)
    b = [
        (x0 + x1) / 2 / size_w,
        (y0 + y1) / 2 / size_h,
        (x1 - x0) / size_w,
        (y1 - y0) / size_h,
    ]
    return YoloBoxes(torch.stack(b, dim=-1))


def coco_to_pascal(coco: CoCoBoxes) -> PascalBoxes:
    x0, y0, w, h = coco.unbind(-1)
    b = [x0, y0, x0 + w, y0 + h]
    return PascalBoxes(torch.stack(b, dim=-1))


def yolo_to_pascal(yolo: YoloBoxes, size: Size) -> PascalBoxes:
    cx, cy, w, h = yolo.unbind(-1)
    size_w, size_h = size
    b = [
        (cx - 0.5 * w) * size_w,
        (cy - 0.5 * h) * size_h,
        (cx + 0.5 * w) * size_w,
        (cy + 0.5 * h) * size_h,
    ]
    return PascalBoxes(torch.stack(b, dim=-1))


def yolo_to_coco(yolo: YoloBoxes, size: Size) -> CoCoBoxes:
    x0, y0, x1, y1 = yolo_to_pascal(yolo, size).unbind(-1)
    b = torch.stack([x0, y0, x1 - x0, y1 - y0], dim=-1).long()
    return CoCoBoxes(b)
