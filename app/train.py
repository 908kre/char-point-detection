import random
import numpy as np
import torch
from app.models.centernet import (
    CenterNet as NNModel,
    Criterion,
)
from app import config
from torch import nn
from torch.utils.data import DataLoader



class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module,
    ) -> None:
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
