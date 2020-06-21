import typing as t
import random
import numpy as np
import torch
from app.models.centernet import (
    CenterNet as NNModel,
    Criterion,
    PreProcess,
)
from app import config
from app.utils import ModelLoader
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_loader: ModelLoader,
    ) -> None:
        self.device = torch.device(config.device)
        self.model_loader = model_loader
        self.model = model_loader.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.train_loader = train_loader
        self.criterion = Criterion()
        self.preprocess = PreProcess()

    def train(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            self.train_one_epoch()

    def train_one_epoch(self) -> None:
        self.model.train()
        epoch_loss = 0
        count = 0
        loader = self.train_loader
        for samples, targets, ids in loader:
            print(samples, targets, ids)
            #  count += 1
            #  samples, cri_targets = self.preprocess((samples, targets))
            #  outputs = self.model(samples)
            #  loss = self.criterion(outputs, cri_targets)
            #  self.optimizer.zero_grad()
            #  loss.backward()
            #  self.optimizer.step()
