import typing as t
from sklearn import preprocessing
import numpy as np
from .entities import Category, Floor, Quater
import re


def encode_label(series: t.Any) -> t.Any:
    le = preprocessing.LabelEncoder()
    return le.fit_transform(series)


def parse_age(value: str) -> t.Optional[int]:
    res = re.search(r"\d+", str(value), 0)
    if res is not None:
        number = int(res.group())
        if "昭和" in value:
            return 1900 + number + 25
        elif "平成" in value:
            return 2000 + number - 12
        else:
            raise Exception(f"Cannot parse {value}")
    return None


def parse_floor(value: str) -> t.Optional[Floor]:
    if not isinstance(value, str):
        return None
    row: Floor = {
        "dinning": 0,
        "living": 0,
        "room": 0,
        "kitchen": 0,
        "storage": 0,
    }
    res = re.search(r"\d+", str(value), 0)
    if value == "スタジオ":
        row["room"] = 1

    # TODO
    if value == "オープンフロア":
        return None

    if res is not None:
        number = int(res.group())
        row["room"] = number

    if "K" in value:
        row["kitchen"] = 1

    if "S" in value:
        row["storage"] = 1

    if "L" in value:
        row["living"] = 1

    if "D" in value:
        row["dinning"] = 1
    return row


def parse_quater(value: str) -> t.Optional[Quater]:
    if not isinstance(value, str):
        return None
    res = re.findall(r"\d+", str(value))
    if len(res) < 2:
        return None
    row: Quater = {
        "year": int(res[0]),
        "quarter": int(res[1]),
    }
    return row


def parse_erea(value: str) -> t.Optional[int]:
    if not isinstance(value, str):
        return None
    res = re.findall(r"\d+", str(value))
    if len(res) > 0:
        return int(res[0])
    return None
