import torch
from app.models.efficent_det import ClipBoxes, BBoxTransform


def test_clip_boxes() -> None:
    images = torch.ones((1, 1, 10, 10))
    boxes = torch.Tensor([[[14, 0, 20, 0]]])
    fn = ClipBoxes()
    res = fn(boxes, images)

    assert (
        res - torch.Tensor([[[14, 0, 10, 0]]])
    ).sum() == 0  # TODO ??? [10, 0, 10, 0]


def test_bbox_transform() -> None:
    ...
