import cv2
import torch
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
from torch.utils.data import Dataset
import albumentations as albm
from .common import imread
from ..transforms import RandomDilateErode, RandomLayout, RandomRuledLines
from albumentations.pytorch.transforms import ToTensorV2
from object_detection.entities import (
    CoCoBoxes,
    TrainSample,
    Image,
    Labels,
    YoloBoxes,
    ImageId,
    coco_to_yolo,
)


class NegativeDataset(Dataset):
    def __init__(
        self, image_dir: str, max_size: int, transforms: Optional[Callable] = None
    ) -> None:
        self.image_files = sorted(Path(image_dir).glob("*.jpg"))
        assert len(self.image_files) > 0
        self.preprocess = albm.Compose(
            [
                albm.ShiftScaleRotate(rotate_limit=10),
                albm.PadIfNeeded(
                    min_width=max_size,
                    min_height=max_size,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                albm.RandomCrop(max_size, max_size),
                RandomDilateErode(ks_limit=(1, 3)),
            ],
            bbox_params={"format": "coco", "label_fields": ["labels1", "labels2"]},
        )
        self.transforms = transforms
        self.postprocess = ToTensorV2()

    def __getitem__(self, idx: int) -> TrainSample:
        image_file = self.image_files[idx]
        sample = dict(
            image=imread(str(image_file))[..., ::-1],
            source="negative",
            image_id=image_file.stem,
            bboxes=np.zeros((0, 4), dtype=float),  # no box
            labels1=np.zeros((0,), dtype=int),
            labels2=np.zeros((0,), dtype=int),
        )
        w, h = tuple(sample["image"].shape[:2][::-1])
        sample = self.preprocess(**sample)
        if self.transforms is not None:
            sample = self.transforms(sample)
        image = self.postprocess(image=sample["image"] / 255.0)["image"]
        boxes = coco_to_yolo(CoCoBoxes(torch.tensor(sample["bboxes"])), (w, h))
        return (
            ImageId(sample["image_id"]),
            Image(image.float()),
            boxes,
            Labels(torch.tensor(sample["labels1"])),
        )

    def __len__(self) -> int:
        return len(self.image_files)
