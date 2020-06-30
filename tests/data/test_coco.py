import torch
import numpy as np
from app.data.coco import CocoDataset
from object_detection.utils import DetectionPlot


def test_CocoDataset() -> None:
    dataset = CocoDataset(
        image_dir="/store/datasets/preview",
        annot_file="/store/datasets/preview/20200611_coco_imglab.json",
        max_size=512,
    )
    for i in range(10):
        id, image, boxes, _ = dataset[0]
        _, h, w = image.shape
        plot = DetectionPlot(figsize=(10, 10), w=w, h=h)
        plot.with_image(image)
        plot.with_yolo_boxes(boxes)
        plot.save(f"/store/tests/plot-coco-{i}.png")
