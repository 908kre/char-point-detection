import pytest
from app.dataset import Dataset
from app.entities import Annotations
import typing as t
from app.cache import Cache
from app.preprocess import load_labels, get_annotations

cache = Cache("/store/tmp")


def test_dataset() -> None:
    labels = cache("labels", load_labels)("/store/dataset/labels.csv")
    annotations = cache("train_annotations", get_annotations)(
        "/store/dataset/train.csv", labels
    )

    d = Dataset(annotations)
    assert len(d) == 142119
