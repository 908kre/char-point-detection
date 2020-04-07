from pathlib import Path
import typing as t
from dask.multiprocessing import get

import joblib
import pandas as pd
from cytoolz.curried import map, pipe
from matplotlib.pyplot import savefig
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from .dataset import Dataset
from .cache import Cache
from sklearn.model_selection import TimeSeriesSplit
from .preprocess import load_labels, get_annotations, get_summary

cache = Cache("/store/tmp")


def eda() -> t.Any:
    labels = cache("labels", load_labels)("/store/dataset/labels.csv")
    print(f"{len(labels)=}")
    train_annotations = cache("train_annotations", get_annotations)(
        "/store/dataset/train.csv", labels
    )
    train_summary = cache("train_summary", get_summary)(train_annotations, labels)
    print(f"{train_summary=}")


def train() -> t.Any:
    labels = cache("labels", load_labels)("/store/dataset/labels.csv")
    train_annotations = cache("train_annotations", get_annotations)(
        "/store/dataset/train.csv", labels
    )
