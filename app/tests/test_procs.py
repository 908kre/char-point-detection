import typing as t
from app.preprocess import load_labels, encode_attribute, get_summary, get_annotations
import numpy as np
import pandas as pd
from app.cache import Cache

cache = Cache("/store/tmp")


def test_load_lablels() -> None:
    labels = load_labels("/store/dataset/labels.csv")
    assert len(labels) == 3474


def test_encode_attribute() -> None:
    res = encode_attribute("tags::zodiac")
    assert ("tags", "zodiac") == res


def test_get_annotations() -> None:
    labels = cache("labels", load_labels)("/store/dataset/labels.csv")
    res = get_annotations("/store/dataset/train.csv", labels)
    print(res)
