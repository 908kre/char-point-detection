from pathlib import Path
import typing as t
import matplotlib.pyplot as plt
from app.train import Trainer
from app.preprocess import load_lables, KFold


def eda() -> t.Any:
    ...


def train(fold_idx: int) -> None:
    images = load_lables()
    kf = KFold()
    train_data, test_data = list(kf(images))[fold_idx]
    trainer = Trainer(train_data, test_data)
    trainer.train(1)
