import typing as t
from torch import Tensor

CoCoBoxes = t.NewType("CoCoBoxes", Tensor)
Labels = t.NewType("Labels", Tensor)
Confidences = t.NewType("Confidences", Tensor)


class PredBoxes:
    boxes: CoCoBoxes
    confidences: Confidences

    def __init__(self, boxes: CoCoBoxes, confidences: Confidences) -> None:
        self.boxes = boxes
        self.confidences = confidences


class LabelBoxes:
    boxes: CoCoBoxes
    labels: Labels

    def __init__(self, boxes: CoCoBoxes, labels: Labels) -> None:
        self.boxes = boxes
        self.labels = labels
