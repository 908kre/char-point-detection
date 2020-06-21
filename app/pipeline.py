from cytoolz.curried import groupby, valmap, pipe, unique, map, reduce
from pathlib import Path
import typing as t
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from .data.coco import CocoDataset
from torch.utils.data import DataLoader, Subset
from app.models.centernet import collate_fn, CenterNet
from app.utils import ModelLoader
from app import config
from app.train import Trainer
from app.preprocess import kfold


def train(fold_idx: int) -> None:
    dataset = CocoDataset(
        image_dir="/store/datasets/preview",
        annot_file="/store/datasets/preview/20200611_coco_imglab.json",
    )
    fold_keys = [len(dataset[i][2]) // 20 for i in range(len(dataset))]
    train_idx, test_idx = list(kfold(n_splits=config.n_splits, keys=fold_keys))[
        fold_idx
    ]
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=config.batch_size,
        drop_last=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=config.batch_size,
        drop_last=True,
        collate_fn=collate_fn,
    )
    model_loader = ModelLoader(out_dir="/store/{fold_idx,}", model=CenterNet())

    trainer = Trainer(train_loader, test_loader, model_loader,)
    trainer.train(100)


def submit() -> None:
    ...
