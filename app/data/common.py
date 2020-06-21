import typing as t
import os
import numpy as np
import cv2
import torch
from torch import Tensor
from torch.utils.data import Dataset


T = t.TypeVar("T")


class Transformed(Dataset):
    def __init__(self, dataset: Dataset, transforms: t.Callable[[t.Any], T]) -> None:
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx: int) -> T:
        return self.transforms(self.dataset[idx])

    def __len__(self) -> int:
        return len(self.dataset)


def collate_fn(samples: t.Any) -> t.Dict[str, t.Any]:
    keys = list(samples[0].keys())
    batch = {}
    for key in keys:
        if not all((key in _) for _ in samples):
            continue
        values: t.Any = [_[key] for _ in samples]
        if all(isinstance(_, list) for _ in values):
            values = [torch.tensor(_) for _ in values]
        elif all(isinstance(_, np.ndarray) for _ in values):
            values = [torch.from_numpy(_) for _ in values]
        if all(isinstance(_, torch.Tensor) for _ in values):
            shape = values[0].shape
            if all((_.shape == shape) for _ in values):
                values = torch.stack(values)
        batch[key] = values
    return batch


def imread(
    filename: str, flags: t.Any = cv2.IMREAD_COLOR, dtype: t.Any = np.uint8
) -> t.Any:
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite(filename: str, img: t.Any, params: t.Optional[t.Any] = None) -> bool:
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode="w+b") as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def coco_to_pascal(boxes: Tensor) -> Tensor:
    boxes = boxes.clone()
    boxes[..., 2:] += boxes[..., :2]
    return boxes
