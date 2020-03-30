import typing as t
from app.preprocess import load_labels, encode_attribute, get_summary, get_annotations
import numpy as np
import pandas as pd


def test_markov_chain() -> None:
    load_labels("/store/dataset/labels.csv")


def test_encode_attribute() -> None:
    res = encode_attribute("tags::zodiac")
    assert ("tags", "zodiac") == res

    #  res = get_summary(
    #      {
    #          1: {"id": 1, "category": "culture", "detail": "aaa",},
    #          2: {"id": 2, "category": "media", "detail": "bbb",},
    #      }
    #  )
    #  print(res)
