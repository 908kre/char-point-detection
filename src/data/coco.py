from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from .common import imread


class CocoDataset(Dataset):

    def __init__(self, image_dir:str, annot_file:str, transforms=None):
        self.image_dir = image_dir
        self.annot_file = Path(annot_file)
        self.coco = COCO(self.annot_file)
        self.image_ids = sorted(self.coco.imgs.keys())
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        bboxes = np.stack([
            _["bbox"]
            for _ in sorted(self.coco.imgToAnns[image_id], key=lambda _: _["id"])
        ])
        image = imread(str(self.image_dir / self.coco.imgs[image_id]["file_name"]))[..., ::-1]
        return {
            "image_id": image_id,
            "image": image,
            "bboxes": bboxes
        }

    def __len__(self):
        return len(self.image_ids)
