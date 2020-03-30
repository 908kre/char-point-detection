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
from .trans import load_labels

cache = Cache("/store/tmp")


def eda() -> t.Any:
    tags = cache("labels", load_labels)("/store/dataset/labels.csv")
    print(f"{tags[0]=}")
