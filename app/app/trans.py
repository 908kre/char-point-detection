import typing as t
import pandas as pd
from .entities import Label, Labels
import seaborn as sns

sns.set()


def load_labels(path: str) -> Labels:
    df = pd.read_csv(path)
    rows: t.Dict[int, Label] = dict()
    for (idx, value) in df.iterrows():
        c, d = encode_attribute(value["attribute_name"])
        rows[idx] = {"id": idx, "category": c, "detail": d}
    return rows


def load_images(path: str, labels: Labels) -> t.Any:
    df = pd.read_csv(path)
    rows: t.Dict[int, Label] = dict()
    for (idx, value) in df.iterrows():
        c, d = encode_attribute(value["attribute_name"])
        rows[idx] = {"id": idx, "category": c, "detail": d}
    return rows


def encode_attribute(name: str) -> t.Tuple[str, str]:
    splited = name.split("::")
    return splited[0], splited[1]
