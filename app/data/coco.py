import typing as t
from pathlib import Path
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from .common import imread
from object_detection.entities import (
    CoCoBoxes,
    TrainSample,
    Image,
    Labels,
    YoloBoxes,
    ImageId,
)
from object_detection.entities.box import coco_to_yolo
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as albm
from cytoolz.curried import map, pipe, concat


class CocoDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        annot_file: str,
        max_size: int,
        transforms: t.Callable = None,
        mode: "str" = "train",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.annot_file = Path(annot_file)
        self.coco = COCO(self.annot_file)
        self.image_ids = sorted(self.coco.imgs.keys())
        self.mode = mode
        bbox_params = {"format": "coco", "label_fields": ["labels"]}
        self.pre_transforms = albm.Compose(
            [
                albm.LongestMaxSize(max_size=max_size),
                albm.PadIfNeeded(
                    min_width=max_size,
                    min_height=max_size,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ],
            bbox_params=bbox_params,
        )
        self.post_transforms = albm.Compose([ToTensorV2(),])

    def __getitem__(self, idx: int) -> TrainSample:
        image_id = self.image_ids[idx]
        boxes = np.stack(
            [
                _["bbox"]
                for _ in sorted(self.coco.imgToAnns[image_id], key=lambda _: _["id"])
            ]
        )
        areas = boxes[:, 2] * boxes[:, 3]
        image_path = self.image_dir / self.coco.imgs[image_id]["file_name"]
        image_name = image_path.stem
        image = (imread(str(image_path)) / 255).astype(np.float32)
        boxes = boxes[areas > 0.0]
        labels = np.zeros(boxes.shape[:1])

        res = self.pre_transforms(image=image, bboxes=boxes, labels=labels)
        image = res["image"]
        boxes = CoCoBoxes(torch.tensor(res["bboxes"]))
        image = self.post_transforms(image=image)["image"]
        _, h, w = image.shape
        boxes = coco_to_yolo(boxes, (w, h))
        labels = torch.zeros(boxes.shape[:1])
        return (
            ImageId(image_name),
            Image(image.float()),
            YoloBoxes(boxes.float()),
            Labels(labels),
        )

    def __len__(self) -> int:
        return len(self.image_ids)
