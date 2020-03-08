import typing as t
from sklearn import preprocessing
import numpy as np
from .entities import Category, Floor, Quater
from pathlib import Path
import pickle
import re


def label_encode(series: t.Any, cache_dir: str, key: str) -> t.Any:
    series = series.fillna("NaN")
    le = preprocessing.LabelEncoder()
    fpath = Path(cache_dir).joinpath(key)
    if fpath.exists():
        with open(fpath, "rb") as f:
            le.classes_ = pickle.load(f)
    le = le.fit(series)
    with open(fpath, "wb") as f:
        pickle.dump(le.classes_, f)
    return le.transform(series)


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


def parse_floor(value: str) -> Floor:
    if not isinstance(value, str) or (value == "オープンフロア"):
        return {
            "dinning": None,
            "living": None,
            "room": None,
            "kitchen": None,
            "storage": None,
        }

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


def parse_duration(value: str) -> t.Optional[int]:
    if not isinstance(value, str):
        return None
    res = re.findall(r"\d+H", str(value))
    if len(res) > 0:
        res = re.findall(r"\d+", res[-1])
        if len(res) > 0:
            return int(res[-1]) * 60

    res = re.findall(r"\d+", str(value))
    if len(res) > 0:
        return int(res[0])
    return None


def fillna_mean(series: t.Any, *kwargs: t.Any) -> t.Any:
    return series.fillna(series.mean())
