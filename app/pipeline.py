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
from pprint import pprint


cache = Cache("/store/tmp")


def eda() -> t.Any:
    ...


def train() -> t.Any:
    ...
