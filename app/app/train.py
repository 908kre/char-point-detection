from .models import UNet
from .dataset import Dataset
import os
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

device = torch.device("cuda")

def train(dataset: Dataset,) -> None:
    train_loader = DataLoader(
        dataset, batch_size=64, drop_last=True, shuffle=True, num_workers=16
    )
    model = UNet(in_channels=1,).to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4,
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, 0.0001)
    criterion = nn.MSELoss()

    for e in range(1000):
        model.train()
        running_loss = 0.0
        for x_batch, label_batch in train_loader:
            x_batch = x_batch.to(device).float()
            label_batch = label_batch.to(device).float()
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                y_batch = model(x_batch)
                loss = criterion(y_batch, label_batch)
                print(loss)
                loss.backward()
                optimizer.step()
        lr_scheduler.step(e)
