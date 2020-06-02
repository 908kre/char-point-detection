import typing as t
from torch.utils.data import DataLoader
import torch
from logging import getLogger


#  from app.models import NNModel
from app.entities import Images
from app.dataset import TrainDataset, collate_fn

logger = getLogger(__name__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DataLoaders = t.TypedDict("DataLoaders", {"train": DataLoader, "test": DataLoader,})


class Trainer:
    def __init__(self, train_data: Images, test_data: Images,) -> None:
        #  self.model = NNModel().to(device)
        #  self.optimizer = torch.optim.AdamW(self.model.parameters(),)
        self.data_loaders: DataLoaders = {
            "train": DataLoader(
                TrainDataset(train_data),
                shuffle=True,
                batch_size=32,
                drop_last=True,
                collate_fn=collate_fn,
            ),
            "test": DataLoader(
                TrainDataset(test_data),
                shuffle=True,
                batch_size=1,
                collate_fn=collate_fn,
            ),
        }

    def train(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            self.train_one_epoch()
            logger.info(f"{epoch=}")

    def train_one_epoch(self) -> None:
        for images, targets, image_ids in self.data_loaders["train"]:
            ...
