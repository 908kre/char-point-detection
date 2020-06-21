from cytoolz.curried import groupby, valmap, pipe, unique, map, reduce
from pathlib import Path
import typing as t
import matplotlib.pyplot as plt
from .data.coco import CocoDataset
from app import config
from app.train import Trainer


def train(fold_idx: int) -> None:
    ...

    #  trainer = Trainer()
    #  trainer.train(100)


def submit() -> None:
    ...
