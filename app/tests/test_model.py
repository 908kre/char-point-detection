import pytest
from app.models import SCSEModule, SENeXt, SENextBottleneck
import torch
from torchsummary import summary


def test_scse() -> None:
    x = torch.randn(32, 256, 320, 320)
    model = SCSEModule(in_channels=256, reduction=12,)
    y = model(x)
    assert x.shape == y.shape


def test_senextbottleneck() -> None:
    in_channels = 1024
    out_channels = 3474
    h = 128
    w = 128
    x = torch.randn(1, in_channels, h, w)
    model = SENextBottleneck(in_channels=in_channels, out_channels=out_channels,)
    y = model(x)
    assert y.shape == (1, out_channels, h, w)


def test_senext() -> None:
    model = SENeXt(in_channels=3, out_channels=3474, depth=3, width=64, ratio=2).to(
        "cuda"
    )
    summary(model, (3, 128, 128))
