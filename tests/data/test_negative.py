import pytest
import torch
import numpy as np
from app.data.negative import NegativeDataset
from object_detection.utils import DetectionPlot


@pytest.mark.parametrize("max_size,", [512, 1024,])
def test_negative(max_size: int) -> None:
    dataset = NegativeDataset(image_dir="/store/negative/images", max_size=max_size)
    for i in range(10):
        id, image, boxes, _ = dataset[0]
        _, h, w = image.shape
        plot = DetectionPlot(figsize=(10, 10), w=w, h=h)
        plot.with_image(image)
        plot.with_yolo_boxes(boxes)
        plot.save(f"/store/tests/plot-negative-{max_size}-{i}.png")
