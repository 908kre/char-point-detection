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
    for i in range(len(dataset)):
        id, image, boxes, _ = dataset[i]
        _, h, w = image.shape
        assert h == max_size
        assert w == max_size
        plot = DetectionPlot(figsize=(10, 10), w=w, h=h)
        plot.with_image(image)
        plot.with_yolo_boxes(boxes)
        plot.save(f"/store/tests/plot-coco-{max_size}-{i}.png")
