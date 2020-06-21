import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import cv2
from torch.utils.data import Dataset
import albumentations as albm
from .imagefile import ImageFileDataset
from ..transforms import RandomDilateErode, RandomLayout, RandomRuledLines


class KuzushijiDataset(Dataset):
    def __init__(self, image_dir: str, annot_file: str, transforms=None):
        self.image_dataset = ImageFileDataset(image_dir, ".jpg")
        self.annot_file = annot_file
        self.preprocess = albm.Compose(
            [
                RandomDilateErode(ks_limit=(1, 5)),
                RandomLayout(800, 800, size_limit=(0.5, 1.0)),
                RandomRuledLines(),
            ],
            bbox_params={"format": "coco", "label_fields": ["labels"]},
        )
        self.transforms = transforms
        titles = [
            Path(_).stem.replace("_", "-").split("-")[0]
            for _ in self.image_dataset.image_ids
        ]
        self.titles = LabelEncoder().fit_transform(titles)
        labels = pd.read_csv(annot_file, index_col=0).iloc[:, 0]
        self.image_ids = labels.index.values
        self.bboxes = [self.decode_label(_) for _ in labels.values]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        sample = self.image_dataset.lookup(image_id + ".jpg")
        sample["image_id"] = image_id
        image_size = sample["image"].shape[:2][::-1]
        bboxes = self.filter_bboxes(self.bboxes[idx].copy(), image_size)
        sample["bboxes"] = bboxes
        sample["labels"] = np.zeros(len(bboxes), dtype=int)  # dummy
        sample = self.preprocess(**sample)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def decode_label(bboxes_str: str) -> np.ndarray:
        tokens = bboxes_str.split(" ")
        bboxes = []
        for idx in range(0, len(tokens), 5):
            bbox = [int(_) for _ in tokens[idx + 1 : idx + 5]]
            bboxes.append(bbox)
        return np.array(bboxes)

    @staticmethod
    def filter_bboxes(bboxes: np.ndarray, image_size, min_area=100) -> np.ndarray:
        eps = 1e-6
        w, h = image_size
        bboxes[:, 2:] += bboxes[:, :2]  # coco to pascal
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, w - eps)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, h - eps)
        bboxes[:, 2:] -= bboxes[:, :2]  # pascal to coco
        area = bboxes[:, 2] * bboxes[:, 3]
        return bboxes[area >= min_area]
