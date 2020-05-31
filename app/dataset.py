import numpy as np
import typing as t
from app.entities import Images
from torch.utils.data import Dataset


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
        image_arr = image.get_arr()
        box_arrs = np.stack([x.to_arr() for x in image.bboxes])

        return {
            "image_id": image.id,
            "image": image_arr,
            "bboxes": box_arrs,
        }
