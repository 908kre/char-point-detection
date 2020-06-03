from pathlib import Path
import typing as t
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from .entities import Images

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


class KuzushijiDataset(Dataset):

    def __init__(self, image_dir: str, annot_file: str):
        self.image_dir = Path(image_dir)
        annot = pd.read_csv(annot_file)
        self.image_ids = annot.image_id.values
        self.bboxes = [self.decode_label(_) for _ in annot.labels.values]

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        image = cv2.imread(str(self.image_dir / f"{image_id}.jpg"))[..., ::-1].copy()
        bboxes = self.bboxes[idx]
        return dict(image_id=idx, image=image, bboxes=bboxes)

    def __len__(self) -> int:
        return len(self.image_ids)

    @staticmethod
    def decode_label(bboxes_str: str) -> np.ndarray:
        tokens = bboxes_str.split(" ")
        bboxes = []
        for idx in range(0, len(tokens), 5):
            bbox = [int(_) for _ in tokens[idx + 1:idx + 5]]
            bboxes.append(bbox)
        return np.array(bboxes)
