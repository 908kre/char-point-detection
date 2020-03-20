from .models import UNet
from .dataset import Dataset
import os
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader


def train(dataset: Dataset,) -> None:
    train_loader = DataLoader(
        dataset, batch_size=32, drop_last=True, shuffle=True, num_workers=16
    )
    model = UNet(in_channels=1,)
    optimizer = optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4,
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, 0.0001)
    criterion = nn.MSELoss()

    for e in range(1000):
        lr_scheduler.step(e)
        model.train()
        running_loss = 0.0
        for x_batch, label_batch in train_loader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                y_batch = model(x_batch.float())
                loss = criterion(y_batch, label_batch.float())
                print(loss)
                loss.backward()
                optimizer.step()
