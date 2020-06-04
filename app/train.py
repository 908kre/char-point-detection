import typing as t
from torch.utils.data import DataLoader
import torch
from logging import getLogger

from app.entities import Images

logger = getLogger(__name__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DataLoaders = t.TypedDict("DataLoaders", {"train": DataLoader, "test": DataLoader,})


class Trainer:
    def __init__(self, train_data: Images, test_data: Images,) -> None:
        ...

    def train(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            self.train_one_epoch()
            logger.info(f"{epoch=}")

    def train_one_epoch(self) -> None:
        ...
