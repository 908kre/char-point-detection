import typing as t
import numpy as np
import torch
from app.models.centernet import CenterNet, SoftHeatMap, ToBoxes
from app.entities import YoloBoxes, Image
from app.entities.box import yolo_to_coco
from app.utils import DetectionPlot


def test_centernet() -> None:
    inputs = torch.rand((1, 3, 1024, 1024))
    fn = CenterNet()
    heatmap, sizemap = fn(inputs)
    assert heatmap.shape == (1, 1, 1024 // 2, 1024 // 2)


def test_softheatmap() -> None:
    in_boxes = YoloBoxes(torch.tensor([[0.2, 0.4, 0.1, 0.3]]))
    in_image = Image(torch.zeros(1, 100, 100))
    to_boxes = ToBoxes(thresold=0.1)
    to_heatmap = SoftHeatMap()
    hm, sm = to_heatmap(in_boxes, in_image)
    assert (hm.eq(1).nonzero()[0, 2:] - torch.tensor([[40, 20]])).sum() == 0  # type: ignore
    assert (sm.nonzero()[0, 2:] - torch.tensor([[40, 20]])).sum() == 0  # type: ignore
    assert hm.shape == (1, 1, 100, 100)
    assert sm.shape == (1, 2, 100, 100)
    assert (sm[0, :, 40, 20] - torch.tensor([0.1, 0.3])).sum() == 0
    out_boxes, _ = next(iter(to_boxes((hm, sm))))
    assert out_boxes[0, 0] == in_boxes[0, 0]
    assert out_boxes[0, 1] == in_boxes[0, 1]
    assert out_boxes[0, 2] == in_boxes[0, 2]
    assert out_boxes[0, 3] == in_boxes[0, 3]
    plot = DetectionPlot()
    plot.with_image(hm[0, 0])
    plot.with_yolo_boxes(in_boxes, color="blue")
    plot.with_yolo_boxes(out_boxes)
    plot.save(f"/store/plot/test-soft-heatmap.png")
