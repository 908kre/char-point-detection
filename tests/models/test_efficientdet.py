import torch
from app.models.efficientdet import (
    ClipBoxes,
    BBoxTransform,
    RegressionModel,
    ClassificationModel,
    EfficientDet,
)


def test_clip_boxes() -> None:
    images = torch.ones((1, 1, 10, 10))
    boxes = torch.tensor([[[14, 0, 20, 0]]])
    fn = ClipBoxes()
    res = fn(boxes, images)

    assert (
        res - torch.tensor([[[14, 0, 10, 0]]])
    ).sum() == 0  # TODO ??? [10, 0, 10, 0]


def test_bbox_transform() -> None:
    boxes = torch.tensor([[[2, 2, 20, 6], [4, 2, 8, 6],]])

    deltas = torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1],]])
    fn = BBoxTransform()
    res = fn(boxes, deltas)


def test_regression_model() -> None:
    images = torch.ones((1, 1, 10, 10))
    fn = RegressionModel(1, num_anchors=9)
    res = fn(images)
    assert res.shape == (1, 900, 4)


def test_classification_model() -> None:
    images = torch.ones((1, 100, 10, 10))
    fn = ClassificationModel(num_features_in=100, num_classes=2)
    res = fn(images)
    assert res.shape == (1, 900, 2)


def test_effdet() -> None:
    images = torch.ones((1, 100, 10, 10))
    annotations = torch.ones((1, 5))
    fn = EfficientDet(num_classes=2)
    res = fn(images, annotations)
    #  assert res.shape == (1, 900, 2)
