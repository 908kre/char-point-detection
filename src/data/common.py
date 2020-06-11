import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class Transformed(Dataset):

    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        return self.transforms(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


def collate_fn(samples):
    keys = list(samples[0].keys())
    batch = {}
    for key in keys:
        values = [_[key] for _ in samples]
        if all(isinstance(_, np.ndarray) for _ in values):
            values = [torch.from_numpy(_) for _ in values]
        if all(isinstance(_, torch.Tensor) for _ in values):
            shape = values[0].shape
            if all((_.shape == shape) for _ in values):
                values = torch.stack(values)
        batch[key] = values
    return batch


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def coco_to_pascal(boxes):
    boxes = boxes.clone()
    boxes[..., 2:] += boxes[..., :2]
    return boxes
