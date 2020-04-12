import typing as t
from app.preprocess import (
    load_labels,
    encode_attribute,
    get_summary,
    get_annotations,
    to_multi_hot,
    evaluate,
)
import numpy as np
import pandas as pd
from app.cache import Cache
from app.entities import Annotations, Annotation

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
    assert len(res) == 142119


def test_to_multhot() -> None:
    annotations: Annotations = [Annotation("a0", [1]), Annotation("a1", [0, 2])]
    res = to_multi_hot(annotations, size=3)
    assert (res != np.array([[0, 1, 0], [1, 0, 1]])).sum() == 0


def test_evaluate() -> None:
    preds = [[0, 1]]
    gts = [[0, 1]]
    res = evaluate(preds, gts)
    assert res == 5 / 9
