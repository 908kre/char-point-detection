from app.models import SCSEModule
import torch


def test_scse() -> None:
    x = torch.randn(32, 256, 320, 320)
    layer = SCSEModule(in_channels=256, reduction=12,)
    y = layer(x)
    assert x.shape == y.shape
