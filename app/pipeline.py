from pathlib import Path
import typing as t
import matplotlib.pyplot as plt
from app.train import Trainer
from app.preprocess import load_lables


def eda() -> t.Any:
    ...


def train(fold_idx: int) -> None:
    images = load_lables()
