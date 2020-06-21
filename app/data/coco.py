import typing as t
from pathlib import Path
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from .common import imread
from app.entities import CoCoBoxes, Sample, Image, Labels, YoloBoxes, ImageId
from app.entities.box import coco_to_yolo
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as albm


transforms = albm.Compose(
    [
        #  albm.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #  albm.ToFloat(max_value=255),
        ToTensorV2(),
    ]
)


class CocoDataset(Dataset):
    def __init__(
        self, image_dir: str, annot_file: str, transforms: t.Callable = None
    ) -> None:
        self.image_dir = Path(image_dir)
        self.annot_file = Path(annot_file)
        self.coco = COCO(self.annot_file)
        self.image_ids = sorted(self.coco.imgs.keys())

    def __getitem__(self, idx: int) -> Sample:
        image_id = self.image_ids[idx]
        bboxes = np.stack(
            [
                _["bbox"]
                for _ in sorted(self.coco.imgToAnns[image_id], key=lambda _: _["id"])
            ]
        )
        areas = bboxes[:, 2] * bboxes[:, 3]
        image_path = self.image_dir / self.coco.imgs[image_id]["file_name"]
        image_name = image_path.stem
        image = (imread(str(image_path)) / 255).astype(np.float32)
        image = transforms(image=image)["image"]
        _, h, w = image.shape
        bboxes = coco_to_yolo(CoCoBoxes(torch.from_numpy(bboxes[areas > 0.0])), (w, h))
        return (
            ImageId(image_name),
            Image(image),
            YoloBoxes(bboxes),
        )

    def __len__(self) -> int:
        return len(self.image_ids)
