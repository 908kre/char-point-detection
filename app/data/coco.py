import typing as t
from pathlib import Path
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from .common import imread, Sample
from app.entities import CoCoBoxes


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
        bboxes = torch.from_numpy(bboxes[areas > 0.0])
        image_name = self.coco.imgs[image_id]["file_name"]
        image = imread(str(self.image_dir / image_name))
        dummy_labels = np.zeros(len(bboxes), dtype=np.int64)
        return Sample(
            id=image_name, image=image, boxes=CoCoBoxes(bboxes), labels=dummy_labels,
        )

    def __len__(self) -> int:
        return len(self.image_ids)
