import numpy as np
import typing as t
from torch.utils.data import Dataset as _Dataset
from skimage import io, transform, color, util
from .entities import Annotations, Annotation

Mode = t.Literal["Test", "Train"]


class Dataset(_Dataset):
    def __init__(self, annotations: Annotations, mode: Mode = "Train", pin_memory:bool=True) -> None:
        self.annotations = annotations
        if mode == "Train":
            self.image_dir = "/store/dataset/train"
        else:
            self.image_dir = "/store/dataset/test"
        self.cache: t.Dict[int, t.Any] = {}
        self.pin_memory = pin_memory

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> t.Tuple[t.Any, t.Any, Annotation]:
        if idx in self.cache:
            return self.cache[idx]

        row = self.annotations[idx]
        img = io.imread(f"{self.image_dir}/{row['id']}.png")
        if len(img.shape) == 2:
            img = color.gray2rgb(img)
        shape = img.shape
        if shape[1] > 300:
            base = shape[1] // 2
            img = img[:, (base - 150) : (base + 150), :]
        if shape[0] > 300:
            base = shape[0] // 2
            img = img[(base - 150) : (base + 150), :, :]
        label = np.zeros(3474)
        for i in row["label_ids"]:
            label[i] = 1
        res = img, label, row
        if self.pin_memory:
            self.cache[idx] = res
        return res
