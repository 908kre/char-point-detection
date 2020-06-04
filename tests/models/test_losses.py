import torch
from app.models.losses import FocalLoss, BBoxIoU


def test_iou() -> None:
    x_bboxes = torch.tensor([[0, 0, 1, 1]]).float()
    y_bboxes = torch.tensor([[0, 0, 2, 2], [2, 2, 4, 4],]).float()

    fn = BBoxIoU()
    res = fn(x_bboxes, y_bboxes)
    assert res.shape == (x_bboxes.shape[0], y_bboxes.shape[0])
    assert res.sum() == torch.tensor((1 + 1) / (4 + 4))


#  def test_focalloss() -> None:

#      fn = FocalLoss()
#      _classes = torch.ones((2, 2))
