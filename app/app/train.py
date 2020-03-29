import numpy as np
import typing as t
from .models import UNet
from .dataset import Dataset
import os
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from mlboard_client import Writer
from datetime import datetime
from .eval import eval
from logging import getLogger
from tqdm import tqdm

logger = getLogger(__name__)


Metrics = t.Dict[str, float]
DEVICE = torch.device("cuda")
writer = Writer("http://192.168.10.8:2020", logger=logger,)
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
    batch_size = 256
    window_size = 1000
    data_loaders: DataLoaders = {
        "train": DataLoader(
            Dataset(train_df, window_size=window_size, stride=1, mode="train",),
            batch_size=batch_size,
            num_workers=16,
            shuffle=True,
            pin_memory=True,
        ),
        "valid": DataLoader(
            Dataset(train_df, window_size=window_size, stride=1, mode="train",),
            batch_size=batch_size,
            num_workers=16,
            shuffle=True,
            pin_memory=True,
        ),
    }
    trainer = Trainer(
        device=DEVICE, data_loaders=data_loaders, objective=nn.CrossEntropyLoss()
    )
    trainer.train(1000)


DataLoaders = t.TypedDict("DataLoaders", {"train": DataLoader, "valid": DataLoader,})


class Trainer:
    def __init__(
        self, objective: t.Any, device: t.Any, data_loaders: DataLoaders
    ) -> None:
        self.device = device
        self.model = UNet(in_channels=1, n_classes=11).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters())
        self.objective = objective
        self.epoch = 1
        self.data_loaders: DataLoaders = data_loaders

    def eval_step(self, data: t.Tuple[t.Any, t.Any]) -> t.Tuple[t.Any, t.Any, t.Any]:
        image, mask = data
        # tta
        output = (
            self.model(image)
            + torch.flip(self.model(torch.flip(image, dims=[2])), dims=[2])
        ) / 2

        loss = self.objective(output, mask)
        pred = F.softmax(output, 1).argmax(dim=1)
        return pred, mask, loss

    def train_step(self, data: t.Any) -> t.Tuple[t.Any, t.Any, t.Any]:
        image, mask = data
        output = self.model(image)
        loss = self.objective(output, mask)
        pred = F.softmax(output, 1).argmax(dim=1)
        return pred, mask, loss

    def train_one_epoch(self) -> None:
        self.model.train()
        epoch_loss = 0.0
        f1_score = 0.0
        for img, msk in tqdm(self.data_loaders["train"]):
            img, msk = img.to(self.device), msk.to(self.device)
            preds, truths, loss = self.train_step((img, msk))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss.item()
            f1_score += eval(
                preds.view(-1).cpu().numpy(), truths.view(-1).cpu().numpy()
            )
        epoch_loss = epoch_loss / len(self.data_loaders["train"])
        f1_score = f1_score / len(self.data_loaders["train"])
        logger.info(f"{epoch_loss=}, {f1_score=}")

    def train(self, max_epochs: int) -> None:
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            self.train_one_epoch()
