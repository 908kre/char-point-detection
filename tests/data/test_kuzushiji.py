import torch
import numpy as np
from app.data.kuzushiji import CodhKuzushijiDataset
from object_detection.utils import DetectionPlot


def test_kuzushiji() -> None:
    print("aa")
    #  print("aa")
    #  dataset = CodhKuzushijiDataset(
    #      image_dir="/store/codh-kuzushiji/resized",
    #      annot_file="/store/codh-kuzushiji/resized/annot.json",
    #  )
    #  for i in range(10):
    #      id, image, boxes, _ = dataset[0]
    #      _, h, w = image.shape
    #      plot = DetectionPlot(figsize=(10, 10), w=w, h=h)
    #      plot.with_image(image)
    #      plot.with_yolo_boxes(boxes)
    #      plot.save(f"/store/tests/plot-coco-{i}.png")
