import torch
import numpy as np
from app.data.coco import CocoDataset
from app.utils import DetectionPlot


def test_CocoDataset() -> None:
    dataset = CocoDataset(
        image_dir="/store/datasets/preview",
        annot_file="/store/datasets/preview/20200611_coco_imglab.json",
    )
    id, image, boxes = dataset[0]
    plot = DetectionPlot(figsize=(10, 10))
    plot.with_image(image)
    plot.with_yolo_boxes(boxes)
    plot.save("/store/tests/plot-coco.png")
