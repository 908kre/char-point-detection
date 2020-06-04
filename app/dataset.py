from pathlib import Path
import typing as t
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from .entities import Images


class KuzushijiDataset(Dataset):
    def __init__(self, image_dir: str, annot_file: str):
        self.image_dir = Path(image_dir)
        annot = pd.read_csv(annot_file)
        self.image_ids = annot.image_id.values
        self.bboxes = [self.decode_label(_) for _ in annot.labels.values]

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        image = cv2.imread(str(self.image_dir / f"{image_id}.jpg"))[..., ::-1].copy()
        bboxes = self.bboxes[idx]
        return dict(image_id=idx, image=image, bboxes=bboxes)

    def __len__(self) -> int:
        return len(self.image_ids)

    @staticmethod
    def decode_label(bboxes_str: str) -> np.ndarray:
        tokens = bboxes_str.split(" ")
        bboxes = []
        for idx in range(0, len(tokens), 5):
            bbox = [int(_) for _ in tokens[idx + 1 : idx + 5]]
            bboxes.append(bbox)
        return np.array(bboxes)
