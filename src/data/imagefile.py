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
from .common import imread

class ImageFileDataset(Dataset):

    def __init__(self, image_dir: str, suffix: str = ".jpg", transforms = None):
        self.image_dir = Path(image_dir)
        self.suffix = suffix
        self.transforms = transforms
        self.image_ids = sorted([
            str(_.relative_to(self.image_dir))
            for _ in self.image_dir.glob("**/*" + self.suffix)
        ])
        self.index = {image_id: idx for idx, image_id in enumerate(self.image_ids)}

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = imread(self.image_dir / image_id)[..., ::-1].copy()
        image_size = tuple(image.shape[:2][::-1])
        sample = dict(image_id=image_id, image=image, image_size=image_size)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def lookup(self, image_id):
        return self.__getitem__(self.index[image_id])

    def __len__(self):
        return len(self.image_ids)
