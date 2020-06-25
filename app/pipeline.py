import torch
import numpy as np
from cytoolz.curried import groupby, valmap, pipe, unique, map, reduce
from pathlib import Path
import typing as t
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from .data.coco import CocoDataset
from .data.random_char import RandomCharDataset
from torch.utils.data import DataLoader, Subset, ConcatDataset
from object_detection.models.centernet import (
    collate_fn,
    CenterNet,
    Trainer,
    Visualize,
    Reg,
)
from object_detection.model_loader import ModelLoader
from app import config
from app.preprocess import kfold


def train(fold_idx: int) -> None:
    dataset = CocoDataset(
        image_dir="/store/datasets/preview",
        annot_file="/store/datasets/preview/20200611_coco_imglab.json",
        max_size=config.max_size,
    )
    #  rc_dataset = RandomCharDataset(max_size=config.max_size, dataset_size=1000,)
    #  dataset: t.Any = ConcatDataset([coco_dataset, rc_dataset])
    fold_keys = [i % 5 for i in range(len(dataset))]
    train_idx, test_idx = list(kfold(n_splits=config.n_splits, keys=fold_keys))[
        fold_idx
    ]
    train_loader = DataLoader(
        Subset(dataset, np.repeat(train_idx, 10)),
        batch_size=config.batch_size,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=config.batch_size,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    out_dir = f"/kaggle/input/models/{fold_idx}"
    model = CenterNet(channels=config.hidden_channels)
    model_loader = ModelLoader(out_dir=f"/store/{fold_idx}", model=model)
    visualize = Visualize("./", "centernet", limit=10, use_alpha=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,)

    trainer = Trainer(
        train_loader, test_loader, model_loader, optimizer, visualize, config.device
    )
    trainer.train(1000)


def submit() -> None:
    ...
