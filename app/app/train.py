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
from concurrent import futures
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
                batch_size=128,
                num_workers=6,
            ),
            "test": DataLoader(
                Dataset(test_data, resolution=128, mode="Test",),
                shuffle=False,
                batch_size=128,
                num_workers=6,
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
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(self.data_loaders["train"])
        epoch = self.epoch
        logger.info(f"{epoch=} train {epoch_loss=}")

    def eval_one_epoch(self) -> None:
        self.model.eval()
        epoch_loss = 0.0
        score = 0.0
        preds:t.Any = []
        labels:t.Any = []
        for img, label in tqdm(self.data_loaders["test"]):
            img, label = img.to(self.device), label.to(self.device)
            with torch.no_grad():
                pred = self.model(img)
                loss = self.objective(pred, label.float())
                epoch_loss += loss.item()
                preds.append(pred.cpu().numpy())
                labels.append(label.cpu().numpy())

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        print(preds.shape)
        print(labels.shape)
        executor = futures.ProcessPoolExecutor()
        thresholds = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
        futs = [
            executor.submit(evaluate, preds, labels, t)
            for t
            in thresholds
        ]
        #  th_scores = {}
        for t, v in zip(thresholds, futures.wait(futs)):
            print(t, v)
            #  th_scores[t] = v

        epoch = self.epoch
        #  threshold, score = max(th_scores.items(), key=lambda x: x[1])
        #  logger.info(f"{epoch=} test {score=} {threshold=}")
        epoch_loss = epoch_loss / len(self.data_loaders["test"])
        logger.info(f"{epoch=} test {epoch_loss=}")

    def train(self, max_epochs: int) -> None:
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            #  self.train_one_epoch()
            self.eval_one_epoch()
