import typing as t


class Label(t.TypedDict):
    id: int
    category: str
    detail: str


Labels = t.Dict[int, Label]
