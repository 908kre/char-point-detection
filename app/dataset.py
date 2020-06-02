import torch
import numpy as np
import typing as t
from app.entities import Images
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2

Row = t.Tuple[t.Any, t.Any, t.Any]
Batch = t.Sequence[Row]


def collate_fn(batch: Batch) -> Row:
    images = []
    boxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        boxes.append(box)
        labels.append(label)
    t_images = torch.stack(images, dim=0)
    return t_images, boxes, labels


class TrainDataset(Dataset):
    def __init__(
        self, images: Images, mode: t.Literal["train", "test"] = "train"
    ) -> None:
        self.rows = list(images.values())
        self.mode = mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> t.Any:
        image = self.rows[index]
        image_arr = ToTensorV2()(image=image.get_arr())["image"]

        box_arrs = torch.from_numpy(np.stack([x.to_arr() for x in image.bboxes]))
        labels = torch.ones((len(box_arrs),))

        return image_arr, box_arrs, labels
