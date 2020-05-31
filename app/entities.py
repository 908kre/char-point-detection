import typing as t


BBox = t.Tuple[int, int, int, int]


class TrainLabel:
    image_id: str
    width: int
    height: int
    bbox: BBox
    source: str

    def __init__(
        self, image_id: str, width: int, height: int, bbox: BBox, source: str
    ) -> None:
        self.image_id = image_id
        self.width = width
        self.height = height
        self.bbox = bbox
        self.source = source
