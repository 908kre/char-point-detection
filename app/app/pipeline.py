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
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from .train import train

sns.set()
cache = Cache("/store/tmp")


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_line(df: pd.DataFrame, path: str) -> str:
    fig, axs = plt.subplots(len(df.columns), 1, sharex=True)

    for i, c in enumerate(df.columns):
        axs[i].plot(df[c])
        axs[i].set_ylabel(c)
    savefig(path)
    plt.close()
    return path


def plot_heatmap(df: pd.DataFrame, path: str) -> str:
    if "open_channels" in df.columns:
        sns.heatmap(df[["signal", "open_channels"]])
    else:
        sns.heatmap(df[["signal"]])
    savefig(path)
    plt.close()
    return path


def filter_df(df: pd.DataFrame, from_idx: int, to_idx: int) -> pd.DataFrame:
    return df.iloc[from_idx:to_idx]


def kfold(df: pd.DataFrame, n_splits: int = 4) -> t.Iterator[t.Tuple[t.Any, t.Any]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train, test in tscv.split(df):
        yield df.iloc[train], df.iloc[test]


def cv(df: t.Any) -> None:
    split = list(kfold(df, 15))[-1]
    train_df, test_df = split
    train(train_df, test_df)


def five_type_remove_drift(train: t.Any) -> t.Any:
    # CLEAN TRAIN BATCH 2
    a = 500000
    b = 600000
    train.loc[train.index[a:b], "signal"] = (
        train.signal[a:b].values - 3 * (train.time.values[a:b] - 50) / 10.0
    )

    def f(x: int, low: float, high: float, mid: float) -> float:
        return -((-low + high) / 625) * (x - mid) ** 2 + high - low

    # CLEAN TRAIN BATCH 7
    batch = 7
    a = 500000 * (batch - 1)
    b = 500000 * batch
    train.loc[train.index[a:b], "signal"] = train.signal.values[a:b] - f(
        train.time[a:b].values, -1.817, 3.186, 325
    )
    # CLEAN TRAIN BATCH 8
    batch = 8
    a = 500000 * (batch - 1)
    b = 500000 * batch
    train.loc[train.index[a:b], "signal"] = train.signal.values[a:b] - f(
        train.time[a:b].values, -0.094, 4.936, 375
    )
    # CLEAN TRAIN BATCH 9
    batch = 9
    a = 500000 * (batch - 1)
    b = 500000 * batch
    train.loc[train.index[a:b], "signal"] = train.signal.values[a:b] - f(
        train.time[a:b].values, 1.715, 6.689, 425
    )
    # CLEAN TRAIN BATCH 10
    batch = 10
    a = 500000 * (batch - 1)
    b = 500000 * batch
    train.loc[train.index[a:b], "signal"] = train.signal.values[a:b] - f(
        train.time[a:b].values, 3.361, 8.45, 475
    )
    return train


def main() -> None:
    executor = ProcessPoolExecutor(max_workers=10)
    train_df = cache("load-train", load)("/store/data/train.csv")
    train_df = cache("remove-drift-train", five_type_remove_drift)(train_df)
    train_df = train_df[0:50000]
    num_chunks = 10
    chunk_size = len(train_df) // num_chunks
    chunks: t.List[t.Any] = []
    for i in range(num_chunks):
        chunk = cache(f"chunk-{i}", filter_df)(
            train_df, i * chunk_size, (i + 1) * chunk_size
        )
        cache(f"plot-line-{i}", plot_line)(chunk, f"/store/plot-line-{i}.png")

    cv(train_df)
