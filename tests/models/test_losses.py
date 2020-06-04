import torch
from app.models.losses import FocalLoss, BBoxIoU
from app.models.anchors import Anchors


def test_iou() -> None:
    x_bboxes = torch.tensor([[0, 0, 1, 1]]).float()
    y_bboxes = torch.tensor([[0, 0, 2, 2], [2, 2, 4, 4],]).float()

    fn = BBoxIoU()
    res = fn(x_bboxes, y_bboxes)
    assert res.shape == (x_bboxes.shape[0], y_bboxes.shape[0])
    assert res.sum() == torch.tensor((1 + 1) / (4 + 4))


def test_focalloss() -> None:
    batch_size = 1
    image = torch.empty(batch_size, 3, 32, 32)
    box_preds = torch.empty((batch_size, 207, 4))
    cls_preds = torch.empty((batch_size, 207, 2))
    annotations = torch.tensor(
        [[[1, 1, 1, 1, 1], [2, 2, 2, 2, -1],]] * batch_size
    ).float()
    anchors = Anchors()(image)

    fn = FocalLoss()
    res = fn(cls_preds, box_preds, anchors, annotations)
    print(res)


#      _classes = torch.ones((2, 2))
