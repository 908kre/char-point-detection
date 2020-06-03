import torch
from app.models.bifpn import BiFPN


def test_bifpn() -> None:
    req = (
        torch.ones((1, 128, 1024 // (2 ** i), 1024 // (2 ** i))) for i in range(3, 8)
    )
    m = BiFPN(channels=128)
    res = m(req)
    #  assert req.shape == res.shape
