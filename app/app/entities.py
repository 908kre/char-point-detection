import typing as t


class Label(t.TypedDict):
    id: str
    category: str
    detail: str


Labels = t.Dict[str, Label]


class Annotation(t.TypedDict):
    id: str
    label_ids: t.Sequence[str]


Annotations = t.Dict[str, Annotation]
