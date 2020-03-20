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
from .train import train

sns.set()
cache = Cache("/store/tmp")


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_line(df: pd.DataFrame, path: str) -> str:
    fig, axs = plt.subplots(len(df.columns), 1, sharex=True)

    for i, c in enumerate(df.columns):
        axs[i].plot(df["time"], df[c])
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


def eda(df: pd.DataFrame) -> None:
    chunk_size = 10000
    num_chunks = len(df) // chunk_size
    chunks: t.List[t.Any] = []
    for i in range(num_chunks):
        chunk = cache(f"chunk-{i}", filter_df)(df, i * chunk_size, (i + 1) * chunk_size)
        cache(f"plot-line-{i}", plot_line)(chunk, f"/store/plot-line-{i}.png")


def main() -> None:
    executor = ProcessPoolExecutor(max_workers=10)
    train_df = cache("load-train", load)("/store/data/train.csv")
    cache("eda-train", eda)(train_df)

    chunk_size = 512 * 100
    dataset = Dataset(train_df, window_size=256, stride=128, mode="train",)
    train(dataset,)
