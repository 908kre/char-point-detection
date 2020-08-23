import pytest
import torch
import numpy as np
from app.data.coco import CocoDataset
from object_detection.utils import DetectionPlot


def test_CocoDataset() -> None:
    max_size = 1024
    dataset = CocoDataset(
        image_dir="/store/datasets/hdata",
        annot_file="/store/datasets/hdata/coco_imglab.json",
        max_size=max_size,
    )
    for i in range(10):
        id, image, boxes, _ = dataset[0]
        _, h, w = image.shape
        plot = DetectionPlot(figsize=(10, 10), w=w, h=h)
        plot.with_image(image)
        plot.with_yolo_boxes(boxes)
        plot.save(f"/store/tests/plot-coco-{max_size}-{i}.png")
