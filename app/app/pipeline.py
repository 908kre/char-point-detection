import typing as t
from dask.multiprocessing import get
import seaborn as sns

sns.set()
import joblib
import pandas as pd
from cytoolz.curried import map, pipe
from matplotlib.pyplot import savefig
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from .cache import Cache


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_line(df: pd.DataFrame, path: str) -> str:
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(df["time"], df["signal"])
    if "open_channels" in df.columns:
        axs[1].plot(df["time"], df["open_channels"])
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


def main() -> None:
    executor = ProcessPoolExecutor(max_workers=10)
    cache = Cache("/store/tmp")
    train_df = cache("load-train", load)("/store/data/train.csv")

    chunk_size = 500
    num_chunks = len(train_df) // chunk_size
    chunks: t.List[t.Any] = []
    for i in range(num_chunks):
        chunk = cache(f"chunk-{i}", filter_df)(
            train_df, i * chunk_size, (i + 1) * chunk_size
        )
        cache(f"plot-line-{i}", plot_line)(chunk, f"/store/plot-line-{i}.png")
