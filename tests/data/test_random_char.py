import torch
import numpy as np
from app.data.random_char import RandomCharDataset
from object_detection.utils import DetectionPlot


def test_RandomCharDataset() -> None:
    dataset = RandomCharDataset(max_size=512)
    for i in range(10):
        id, image, boxes, _ = dataset[i]
        plot = DetectionPlot(figsize=(20, 20))
        plot.with_image(image)
        plot.with_yolo_boxes(boxes, color="red")
        plot.save(f"/store/tests/plot-random-char-{i}.png")

