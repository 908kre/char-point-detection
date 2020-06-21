import typing as t
import numpy as np
import torch
from app.models.centernet import CenterNet
from app.utils import DetectionPlot


def test_centernet() -> None:
    inputs = torch.rand((1, 3, 1024, 1024))
    fn = CenterNet()
    heatmap, sizemap = fn(inputs)
    assert heatmap.shape == (1, 1, 1024 // 2, 1024 // 2)
