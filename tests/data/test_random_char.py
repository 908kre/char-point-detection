import torch
import numpy as np
from app.data.random_char import RandomCharDataset
from app.utils import DetectionPlot


def test_RandomCharDataset() -> None:
    dataset = RandomCharDataset()
    id, image, boxes = dataset[0]
    plot = DetectionPlot(figsize=(10, 10))
    plot.with_image(image)
    plot.with_yolo_boxes(boxes)
    plot.save("/store/tests/plot-random-char.png")
