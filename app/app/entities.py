import typing as t


class Label(t.TypedDict):
    id: int
    category: str
    detail: str


Labels = t.Dict[int, Label]


class Annotation(t.TypedDict):
    id: str
    label_ids: t.Sequence[int]


Annotations = t.List[Annotation]
