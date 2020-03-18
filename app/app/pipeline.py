from dask.multiprocessing import get
import pandas as pd
from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_plot(df: pd.DataFrame, path: str) -> str:
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(df["time"], df["signal"])
    if "open_channels" in df.columns:
        axs[1].plot(df["time"], df["open_channels"])
    savefig(path)
    return path


dsk = {
    "train-load": (load, "/store/train.csv"),
    "test-load": (load, "/store/test.csv"),
    "train-plot": (save_plot, "train-load", "/store/train-plot.png"),
    "test-plot": (save_plot, "test-load", "/store/test-plot.png"),
}


def main() -> None:
    get(dsk, ["train-plot", "test-plot"])
