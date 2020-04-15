from pathlib import Path
import typing as t
from dask.multiprocessing import get

import joblib
import pandas as pd
from cytoolz.curried import map, pipe
from matplotlib.pyplot import savefig
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from .cache import Cache
from .train import Trainer
from .dataset import Dataset
from sklearn.model_selection import TimeSeriesSplit
from .preprocess import (
    load_labels,
    get_annotations,
    get_summary,
    get_images_summary,
    kfold,
    to_multi_hot,
)
from pprint import pprint


cache = Cache("/store/tmp")


def eda() -> t.Any:
    labels = cache("labels", load_labels)("/store/dataset/labels.csv")
    print(f"{len(labels)=}")
    train_annotations = cache("train_annotations", get_annotations)(
        "/store/dataset/train.csv", labels
    )
    train_summary = cache("train_summary", get_summary)(train_annotations, labels)
    print(f"{train_summary=}")
    train_image_summary = cache("train_image_summary", get_images_summary)(
        "/store/dataset/train"
    )
    print(f"{train_image_summary=}")
    test_image_summary = cache("test_image_summary", get_images_summary)(
        "/store/dataset/test"
    )
    print(f"{test_image_summary=}")


def train() -> t.Any:
    labels = cache("labels", load_labels)("/store/dataset/labels.csv")
    train_annotations = cache("train_annotations", get_annotations)(
        "/store/dataset/train.csv", labels
    )
    n_splits = 6
    kfolded = cache(f"kfolded-{n_splits}", kfold)(n_splits, train_annotations)
    for i, (train_data, test_data) in enumerate(kfolded):
        t = Trainer(
            train_data=train_data, test_data=test_data, model_path=f"/store/model-{i}",
        )
        t.train(1000)
