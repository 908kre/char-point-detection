import random
import typing as t
from pathlib import Path
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from object_detection.entities import Sample, Image, Labels, YoloBoxes, ImageId
from object_detection.entities.box import pascal_to_yolo, PascalBoxes
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as albm
from cytoolz.curried import map, pipe, concat, filter
from random_char_image import TextRepo, RandomImage
from glob import glob

ignore_chars = [",", "》", "《", ">", "《", "、", "。"]


class RandomCharDataset(Dataset):
    def __init__(
        self, mode: "str" = "train", max_size: int = 1024, dataset_size: int = 100
    ) -> None:
        self.dataset_size = dataset_size
        self.text = TextRepo().with_file("/store/texts/hvt.txt")
        self.hw_fonts = list(glob("/store/hw_fonts/*.ttf"))
        self.pc_fonts = list(glob("/store/pc_fonts/*.ttf"))

        bbox_params = {"format": "pascal_voc", "label_fields": ["labels"]}
        self.max_size = max_size
        self.post_transforms = albm.Compose([ToTensorV2(),])
        self.pre_transforms = albm.Compose(
            [
                albm.LongestMaxSize(max_size=max_size),
                albm.PadIfNeeded(
                    min_width=max_size,
                    min_height=max_size,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ],
            bbox_params=bbox_params,
        )

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> Sample:
        ri = (
            RandomImage()
            .with_config(
                fontsize=random.randint(14, 32),
                line_space=random.randint(14, 32),
                char_space=random.randint(1, 5),
                direction=random.choice(["row", "column"]),
            )
            .with_text(self.text)
            .with_label_font(random.choice(self.hw_fonts), label=0, is_random=False)
            .with_label_font(random.choice(self.pc_fonts), label=1, is_random=False)
        )
        image, _boxes, labels, chars = ri.get()
        image = (np.array(image) / 255).astype(np.float32)
        boxes = pipe(
            zip(_boxes, labels, chars),
            filter(lambda x: x[1] == 0),
            filter(lambda x: x[2] not in ignore_chars),
            map(lambda x: x[0]),
            list,
            np.array,
        )
        labels = np.zeros((len(_boxes),))
        res = self.pre_transforms(image=image, bboxes=boxes, labels=labels)
        image = res["image"]
        image = self.post_transforms(image=image)["image"]
        _, h, w = image.shape
        pascal_boxes = PascalBoxes(torch.tensor(res["bboxes"]))
        yolo_boxes = pascal_to_yolo(pascal_boxes, (w, h))
        return (
            ImageId(""),
            Image(image.float()),
            YoloBoxes(yolo_boxes.float()),
        )
