import numpy as np
import typing as t
import os
from .entities import Annotations
from .dataset import Dataset
import os
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from mlboard_client import Writer
from datetime import datetime
from .preprocess import evaluate
from .models import SENeXt, FocalLoss
from logging import getLogger
from tqdm import tqdm
from torchvision.transforms import ToTensor
from albumentations.augmentations.transforms import RandomResizedCrop, HorizontalFlip

#
logger = getLogger(__name__)
DEVICE = torch.device("cuda")
DataLoaders = t.TypedDict("DataLoaders", {"train": DataLoader, "test": DataLoader,})


class Trainer:
    def __init__(
        self, train_data: Annotations, test_data: Annotations, model_path: str
    ) -> None:
        self.device = DEVICE
        self.model = SENeXt(in_channels=3, out_channels=3474, depth=2, width=64).to(
            DEVICE
        )
        self.optimizer = optim.Adam(self.model.parameters())
        self.objective = FocalLoss()
        self.epoch = 1
        self.model_path = model_path
        self.data_loaders: DataLoaders = {
            "train": DataLoader(
                Dataset(train_data, resolution=128, mode="Train",),
                shuffle=True,
                batch_size=32,
            ),
            "test": DataLoader(
                Dataset(test_data, resolution=128, mode="Test",),
                shuffle=False,
                batch_size=32,
            ),
        }
        train_len = len(train_data)
        logger.info(f"{train_len=}")
        test_len = len(test_data)
        logger.info(f"{test_len=}")

    def train_one_epoch(self) -> None:
        self.model.train()
        epoch_loss = 0.0
        score = 0.0
        for img, label in tqdm(self.data_loaders["train"]):
            img, label = img.to(self.device), label.to(self.device)
            pred = self.model(img)
            loss = self.objective(pred, label.float())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            score += evaluate((pred > 0.5).int().cpu().numpy(), label.cpu().numpy())
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(self.data_loaders["train"])
        score = score / len(self.data_loaders["train"])
        epoch = self.epoch
        logger.info(f"{epoch=} train {epoch_loss=}")
        logger.info(f"{epoch=} train {score=}")

    def eval_one_epoch(self) -> None:
        self.model.eval()
        epoch_loss = 0.0
        score = 0.0
        for img, label in tqdm(self.data_loaders["test"]):
            img, label = img.to(self.device), label.to(self.device)
            with torch.no_grad():
                pred = self.model(img)
                loss = self.objective(pred, label.float())
                epoch_loss += loss.item()
                score += evaluate((pred > 0.5).int().cpu().numpy(), label.cpu().numpy())
        score = score / len(self.data_loaders["train"])
        epoch_loss = epoch_loss / len(self.data_loaders["test"])
        epoch = self.epoch
        logger.info(f"{epoch=} test {epoch_loss=}")
        logger.info(f"{epoch=} test {score=}")

    def train(self, max_epochs: int) -> None:
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            self.train_one_epoch()
            self.eval_one_epoch()
