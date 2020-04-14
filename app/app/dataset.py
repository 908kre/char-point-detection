import numpy as np
import typing as t
from torch.utils.data import Dataset as _Dataset
from skimage import io, transform, color, util
from .entities import Annotations, Annotation
import cv2
from albumentations.augmentations.transforms import (
    RandomResizedCrop,
    HorizontalFlip,
    CenterCrop,
    Resize,
    PadIfNeeded,
    RandomBrightnessContrast,
    RandomGamma,
    Cutout,
)
from torchvision.transforms import ToTensor


Mode = t.Literal["Test", "Train"]

to_tensor = ToTensor()


class Dataset(_Dataset):
    def __init__(
        self, annotations: Annotations, resolution: int = 320, mode: Mode = "Train",
    ) -> None:
        self.annotations = annotations
        self.mode = mode
        self.image_dir = "/store/dataset/images"
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.annotations)

    def transform(self, img: t.Any) -> t.Any:
        max_hw = max(img.shape[0:2])
        img = PadIfNeeded(max_hw, max_hw, border_mode=cv2.BORDER_REPLICATE)(image=img)[
            "image"
        ]

        if self.mode == "Train":
            img = Cutout(p=0.2)(image=img)["image"]
            img = RandomResizedCrop(self.resolution, self.resolution)(
                image=img
            )["image"]
            img = HorizontalFlip()(image=img)["image"]
            img = RandomBrightnessContrast(p=0.3)(image=img)["image"]
            img = RandomGamma(gamma_limit=(95, 105), p=0.3)(image=img)["image"]

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
