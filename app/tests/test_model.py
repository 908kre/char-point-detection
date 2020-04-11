from app.models import SCSEModule, SEResNeXt
import torch


def test_scse() -> None:
    x = torch.randn(32, 256, 320, 320)
    layer = SCSEModule(in_channels=256, reduction=12,)
    y = layer(x)
    assert x.shape == y.shape

def test_seresnext() -> None:
    x = torch.randn(16, 3, 128, 128)
    layer = SEResNeXt(
        in_channels=3,
        out_channels=3474,
        depth=2,
        width=1024,
    )
    print(layer)
    y = layer(x)
    #  assert x.shape == y.shape
