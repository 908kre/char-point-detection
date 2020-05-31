import typing as t


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


BBoxs = t.List[BBox]


class Image:
    id: str
    width: int
    height: int
    bboxs: BBoxs
    source: str

    def __init__(
        self, id: str, width: int, height: int, bboxs: BBoxs, source: str
    ) -> None:
        self.id = id
        self.width = width
        self.height = height
        self.bboxs = bboxs
        self.source = source


Images = t.Dict[str, Image]
