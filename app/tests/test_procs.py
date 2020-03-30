import typing as t
from app.trans import load_labels, encode_attribute
import numpy as np


def test_markov_chain() -> None:
    rows = load_labels("/store/dataset/labels.csv")
    print(rows)


def test_encode_attribute() -> None:
    res = encode_attribute("tags::zodiac")
    assert ("tags", "zodiac") == res
