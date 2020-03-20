from app.models import DoubleConv, UNet
import torch


def test_double_conv() -> None:
    x = torch.randn(1, 1, 100)
    layer = DoubleConv(in_channels=1, out_channels=12,)
    y = layer(x)
    assert y.size() == (1, 12, 100)

def test_unet() -> None:
    x = torch.randn(1, 1, 256)
    layer = UNet(in_channels=1)
    y = layer(x)
    assert y.size() == x.size()
