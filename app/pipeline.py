from cytoolz.curried import groupby, valmap, pipe, unique, map, reduce
from pathlib import Path
import typing as t
import matplotlib.pyplot as plt
from app.train import Trainer
from app.preprocess import load_lables, KFold, plot_bboxes

def eda_bboxes() -> None:
    images = load_lables()
    bboxes = pipe(
        images.values(),
        map(lambda x: x.bboxes),
        reduce(lambda x, y: x + y),
    )
    plot_bboxes(bboxes, "bbox_sizes.png")


def train(fold_idx: int) -> None:
    images = load_lables()
    kf = KFold()
    train_data, test_data = list(kf(images))[fold_idx]
    trainer = Trainer(train_data, test_data)
    trainer.train(1)
