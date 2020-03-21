from app.models import DoubleConv, UNet, CNet
import torch


def test_double_conv() -> None:
    x = torch.randn(1, 1, 100)
    layer = DoubleConv(in_channels=1, out_channels=12,)
    y = layer(x)
    assert y.size() == (1, 12, 100)


def test_unet() -> None:
    x = torch.randn(1, 1, 256)
    layer = UNet(in_channels=1, n_classes=11)
    y = layer(x)
    assert y.size() == (1, 11, 256)


def test_cnet() -> None:
    x = torch.randn(1, 1, 256)
    layer = CNet(in_channels=1)
    y = layer(x)
    assert y.size() == (1, 1)
