import numpy as np
import typing as t
from torch.utils.data import Dataset as _Dataset
from skimage import io, transform
from .entities import Annotations

Mode = t.Literal["Test", "Train"]


class Dataset(_Dataset):
    def __init__(self, annotations: Annotations, mode: Mode = "Train") -> None:
        self.annotations = annotations
        if mode == "Train":
            self.image_dir = "/store/dataset/train"
        else:
            self.image_dir = "/store/dataset/test"

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> t.Tuple[t.Any, t.Any]:
        row = self.annotations[idx]
        img = io.imread(f"{self.image_dir}/{row['id']}.png")
        label = np.zeros(3474)
        for i in row["label_ids"]:
            label[i] = 1
        return img, label
