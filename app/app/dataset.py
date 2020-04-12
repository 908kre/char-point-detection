import numpy as np
import typing as t
from torch.utils.data import Dataset as _Dataset
from skimage import io, transform, color, util
from .entities import Annotations, Annotation

Mode = t.Literal["Test", "Train"]


class Dataset(_Dataset):
    def __init__(
        self,
        annotations: Annotations,
        mode: Mode = "Train",
        resolution: int = 256,
        pin_memory: bool = False,
    ) -> None:
        self.annotations = annotations
        if mode == "Train":
            self.image_dir = "/store/dataset/train"
        else:
            self.image_dir = "/store/dataset/test"
        self.cache: t.Dict[str, t.Any] = {}
        self.pin_memory = pin_memory
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.annotations)

    def __get_img(self, path: str) -> t.Any:
        if path in self.cache:
            return self.cache[path]

        img = io.imread(path)
        if len(img.shape) == 2:
            img = color.gray2rgb(img)
        shape = img.shape
        half = self.resolution // 2
        if shape[1] > self.resolution:
            base = shape[1] // 2
            img = img[:, (base - half) : (base + half), :]
        if shape[0] > self.resolution:
            base = shape[0] // 2
            img = img[(base - half) : (base + half), :, :]
        img = util.img_as_float(img.transpose((2, 0, 1))).astype(np.float32)
        if self.pin_memory:
            self.cache[path] = img
        return img

    def __getitem__(self, idx: int) -> t.Tuple[t.Any, t.Any, Annotation]:
        row = self.annotations[idx]

        label = np.zeros(3474)
        for i in row["label_ids"]:
            label[i] = 1
        img = self.__get_img(f"{self.image_dir}/{row['id']}.png")
        return img, label, row
