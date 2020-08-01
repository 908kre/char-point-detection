import torch
import numpy as np
from app.data.negative import NegativeDataset
from object_detection.utils import DetectionPlot


def test_negative() -> None:
    dataset = NegativeDataset(image_dir="/store/negative/images",)
    for i in range(10):
        id, image, boxes, _ = dataset[0]
        _, h, w = image.shape
        print(h, w)
        plot = DetectionPlot(figsize=(10, 10), w=w, h=h)
        plot.with_image(image)
        plot.with_yolo_boxes(boxes)
        plot.save(f"/store/tests/plot-negative-{i}.png")
