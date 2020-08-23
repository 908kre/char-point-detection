import torch
import cv2
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
from memory_profiler import profile


class CodhKuzushijiDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        annot_file: str,
        max_size: int,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.annot_file = Path(annot_file)
        with open(annot_file) as fp:
            self.annots = json.load(fp)
        bbox_params={"format": "coco", "label_fields": ["labels"]}
        self.preprocess = albm.Compose(
            [
                albm.OneOf(
                    [
                        albm.ShiftScaleRotate(rotate_limit=5),
                        albm.LongestMaxSize(max_size=max_size),
                    ]
                ),
                RandomLayout(max_size, max_size, (0.5, 1.0)),
                albm.ToGray(p=0.1),
                albm.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.9
                ),
                #  RandomDilateErode(ks_limit=(0.1, 3)),
                albm.Cutout(
                    num_holes=8,
                    max_h_size=max_size // 32,
                    max_w_size=max_size // 32,
                    fill_value=0,
                    p=0.5,
                ),
            ],
            bbox_params=bbox_params,
        )
        self.transforms = transforms
        self.postprocess = albm.Compose([ToTensorV2(),])

    def __getitem__(self, idx: int) -> TrainSample:
        sample = self.annots[idx].copy()
        sample["source"] = "codh"
        image_file = self.image_dir / sample["image_id"]
        sample["image"] = imread(str(image_file))[..., ::-1]
        bboxes = self.filter_bboxes(
            np.array(sample["bboxes"], dtype=float), sample["image"].shape[:2][::-1]
        )
        sample["bboxes"] = bboxes
        sample["labels"] = np.full(len(bboxes), 1, dtype=int)
        sample = self.preprocess(**sample)
        if self.transforms is not None:
            sample = self.transforms(sample)
        image = self.postprocess(image=sample["image"])["image"] / 255.0
        _, h, w = image.shape
        boxes = coco_to_yolo(CoCoBoxes(torch.tensor(sample["bboxes"])), (w, h))
        return (
            ImageId(sample["image_id"]),
            Image(image.float()),
            boxes,
            Labels(torch.tensor(sample["labels"])),
        )

    def __len__(self) -> int:
        return len(self.annots)

    @staticmethod
    def filter_bboxes(
        bboxes: np.ndarray, image_size: Any, min_area: int = 32
    ) -> np.ndarray:
        eps = 1e-6
        w, h = image_size
        bboxes[:, 2:] += bboxes[:, :2]  # coco to pascal
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, w - eps)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, h - eps)
        bboxes[:, 2:] -= bboxes[:, :2]  # pascal to coco
        area = bboxes[:, 2] * bboxes[:, 3]
        return bboxes[area >= min_area]
