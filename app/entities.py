import numpy as np
import typing as t
from skimage.io import imread
from pathlib import Path
from app import config


class BBox:
    x: int
    y: int
    w: int
    h: int

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def to_arr(self,) -> t.Any:
        return np.array([self.x, self.y, self.x + self.w, self.y + self.h])

    def size(self,) -> int:
        return self.w * self.h


BBoxes = t.List[BBox]


class Image:
    id: str
    width: int
    height: int
    bboxes: BBoxes
    source: str

    def __init__(
        self, id: str, width: int, height: int, bboxes: BBoxes, source: str
    ) -> None:
        self.id = id
        self.width = width
        self.height = height
        self.bboxes = bboxes
        self.source = source

    def __repr__(self,) -> str:
        id = self.id
        return f"<Image {id=}>"

    def get_arr(self) -> t.Any:
        image_path = Path(config.image_dir).joinpath(f"{self.id}.jpg")
        return (imread(image_path) / 255).astype(np.float32)


Images = t.Dict[str, Image]
