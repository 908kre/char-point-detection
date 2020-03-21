import numpy as np
import typing as t
from .models import UNet
from .dataset import Dataset
import os
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from mlboard_client import Writer
from datetime import datetime
from .eval import eval
import logging


Metrics = t.Dict[str, float]
DEVICE = torch.device("cuda")
writer = Writer(
    "http://192.168.10.8:2020",
    f"unet-{datetime.now()}",
    {"test": 0},
    logger=logging.getLogger(),
)
SEED = 13
np.random.seed(SEED)
torch.manual_seed(SEED)


def train_epoch(
    data_loader: DataLoader, model: t.Any, optimizer: t.Any, criterion: t.Any,
) -> Metrics:
    running_loss = 0.0
    preds: t.List[int] = []
    labels: t.List[int] = []
    for x_batch, label_batch in data_loader:
        x_batch = x_batch.to(DEVICE).float()
        label_batch = label_batch.to(DEVICE)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            y_batch = model(x_batch)
            loss = criterion(y_batch, label_batch)
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        preds += y_batch.argmax(dim=1).view(-1).cpu().tolist()
        labels += label_batch.int().view(-1).cpu().tolist()
        print("---------------")
        print(preds[-10:])
        print(labels[-10:])
        print("---------------")

    return {"loss": running_loss / len(data_loader), "f1": eval(preds, labels)}


def eval_epoch(data_loader: DataLoader, model: t.Any,) -> Metrics:
    running_loss = 0.0
    model.eval()
    preds: t.List[int] = []
    labels: t.List[int] = []
    for x_batch, label_batch in data_loader:
        x_batch = x_batch.to(DEVICE).float()
        label_batch = label_batch.to(DEVICE).float()
        with torch.set_grad_enabled(False):
            preds += model(x_batch).argmax(dim=1).view(-1).cpu().tolist()
            labels += label_batch.int().view(-1).cpu().tolist()
    return {"f1": eval(preds, labels)}


def train(train_df: t.Any, test_df: t.Any) -> None:
    window_size = len(train_df) // 15
    train_loader = DataLoader(
        Dataset(
            train_df, window_size=window_size, stride=window_size // 3, mode="train",
        ),
        batch_size=2,
        shuffle=True,
    )
    test_loader = DataLoader(
        Dataset(test_df, window_size=window_size, stride=window_size, mode="train"),
        batch_size=2,
    )
    model = UNet(in_channels=1, n_classes=11).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters()
        #  model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4,
    )
    #  lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, 0.0001)
    criterion = nn.CrossEntropyLoss()

    for e in range(1000):
        model.train()
        running_loss = 0.0
        train_metrics = train_epoch(train_loader, model, optimizer, criterion,)
        eval_metrics = eval_epoch(test_loader, model)
        #  lr_scheduler.step(e)
        writer.add_scalars(
            {
                #  "lr": lr_scheduler.get_lr()[0],  # type: ignore
                "train_loss": train_metrics["loss"],
                "eval_f1": eval_metrics["f1"],
                "train_f1": train_metrics["f1"],
            }
        )
