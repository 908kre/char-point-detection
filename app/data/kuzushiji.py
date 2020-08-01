import torch
from pathlib import Path
import json
import numpy as np
from typing import List, Any, Optional, Callable, Tuple
from torch.utils.data import Dataset
import albumentations as albm
from object_detection.entities import (
    CoCoBoxes,
    TrainSample,
    Image,
    Labels,
    YoloBoxes,
    ImageId,
    coco_to_yolo,
)
from albumentations.pytorch.transforms import ToTensorV2
from .common import imread
from ..transforms import RandomDilateErode, RandomLayout, RandomRuledLines


class CodhKuzushijiDataset(Dataset):
    def __init__(self, image_dir: str, annot_file: str, transforms:Optional[Callable]=None) -> None:
        self.image_dir = Path(image_dir)
        self.annot_file = Path(annot_file)
        with open(annot_file) as fp:
            self.annots = json.load(fp)
        self.preprocess = albm.Compose(
            [
                RandomDilateErode(ks_limit=(1, 3)),
                RandomLayout(1024, 1024, size_limit=(0.9, 1.0)),
                RandomRuledLines(),
            ],
            bbox_params={"format": "coco", "label_fields": ["labels1", "labels2"]},
        )
        self.transforms = transforms
        self.postprocess = ToTensorV2()

    def __getitem__(self, idx:int) -> TrainSample:
        sample = self.annots[idx]
        sample["source"] = "codh"
        image_file = self.image_dir / sample["image_id"]
        sample["image"] = imread(str(image_file))[..., ::-1]
        w, h = tuple(sample["image"].shape[:2][::-1])
        bboxes = self.filter_bboxes(np.array(sample["bboxes"]), (w, h))
        sample["bboxes"] = bboxes
        sample["labels1"] = np.full(len(bboxes), 1, dtype=int)
        sample["labels2"] = np.full(len(bboxes), -1, dtype=int)  # dont care
        sample = self.preprocess(**sample)
        if self.transforms is not None:
            sample = self.transforms(sample)
        image = self.postprocess(image=sample['image'] / 255)['image']
        boxes = coco_to_yolo(
            CoCoBoxes(torch.tensor(sample["bboxes"])), (w, h))
        return (
            ImageId(sample["image_id"]),
            Image(image),
            boxes,
            Labels(sample["labels1"]),
        )

    def __len__(self) -> int:
        return len(self.annots)

    @staticmethod
    def filter_bboxes(bboxes: np.ndarray, image_size:Any, min_area:int=32) -> np.ndarray:
        eps = 1e-6
        w, h = image_size
        bboxes[:, 2:] += bboxes[:, :2]  # coco to pascal
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, w - eps)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, h - eps)
        bboxes[:, 2:] -= bboxes[:, :2]  # pascal to coco
        area = bboxes[:, 2] * bboxes[:, 3]
        return bboxes[area >= min_area]
