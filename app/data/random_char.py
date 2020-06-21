import typing as t
from pathlib import Path
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from app.entities import Sample, Image, Labels, YoloBoxes, ImageId
from app.entities.box import pascal_to_yolo, PascalBoxes
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as albm
from cytoolz.curried import map, pipe, concat, filter
from random_char_image import TextRepo, BackgrandRepo, RandomImage
from glob import glob


class RandomCharDataset(Dataset):
    def __init__(self, mode: "str" = "train", length: int = 1024,) -> None:
        self.br = BackgrandRepo().with_file(
            "/store/templates/shiyoukyokashinnseisho09.jpg"
        )
        self.text = TextRepo().with_file("/store/texts/hvt.txt")
        self.ri = (
            RandomImage()
            .with_config(fontsize=28, line_space=30, char_space=20, direction="row")
            .with_backgrand(self.br.get())
            .with_text(self.text)
        )
        for p in glob("/store/hw_fonts/*.ttf"):
            self.ri.with_label_font(p, label=0, is_random=True)

        for p in glob("/store/pc_fonts/*.ttf"):
            self.ri.with_label_font(p, label=1, is_random=False)

        bbox_params = {"format": "pascal_voc", "label_fields": ["labels"]}
        self.length = length
        self.post_transforms = albm.Compose([ToTensorV2(),])
        self.pre_transforms = albm.Compose(
            [
                albm.LongestMaxSize(max_size=length),
                albm.PadIfNeeded(
                    min_width=length,
                    min_height=length,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ],
            bbox_params=bbox_params,
        )

    def __len__(self) -> int:
        return 100

    def __getitem__(self, idx: int) -> Sample:
        image, _boxes, labels = self.ri.get()
        image = (np.array(image) / 255).astype(np.float32)
        _boxes = pipe(
            zip(_boxes, labels), filter(lambda x: x[1] == 0), map(lambda x: x[0]), list
        )
        labels = np.zeros((len(_boxes),))
        _boxes = np.array(_boxes).astype(np.int32)

        res = self.pre_transforms(image=image, bboxes=_boxes, labels=labels)

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
