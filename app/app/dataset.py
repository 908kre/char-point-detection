import numpy as np
import typing as t
from torch.utils.data import Dataset as _Dataset
from skimage import io, transform, color, util
from .entities import Annotations, Annotation
from albumentations.augmentations.transforms import (
    RandomResizedCrop,
    HorizontalFlip,
    CenterCrop,
    Resize,
    PadIfNeeded,
)
from torchvision.transforms import ToTensor


Mode = t.Literal["Test", "Train"]

to_tensor = ToTensor()


class Dataset(_Dataset):
    def __init__(
        self, annotations: Annotations, resolution: int = 256, mode: Mode = "Train",
    ) -> None:
        self.annotations = annotations
        self.mode = mode
        if mode == "Train":
            self.image_dir = "/store/dataset/train"
        else:
            self.image_dir = "/store/dataset/test"
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.annotations)

    def transform(self, img: t.Any) -> t.Any:
        max_hw = max(img.shape[0:2])
        img = PadIfNeeded(max_hw, max_hw)(image=img)["image"]

        if self.mode == "Train":
            img = RandomResizedCrop(self.resolution, self.resolution, scale=(0.5, 1))(
                image=img
            )["image"]
            img = HorizontalFlip()(image=img)["image"]
        else:
            img = Resize(self.resolution, self.resolution)(image=img)["image"]
        img = ToTensor()(img)
        return img

    def __get_img(self, path: str) -> t.Any:

        img = io.imread(path)
        if len(img.shape) == 2:
            img = color.gray2rgb(img)
        img = self.transform(img)
        return img

    def __getitem__(self, idx: int) -> t.Tuple[t.Any, t.Any]:
        row = self.annotations[idx]
        label = np.zeros(3474)
        for i in row.label_ids:
            label[i] = 1
        img = self.__get_img(f"{self.image_dir}/{row.id}.png")
        return img, label
