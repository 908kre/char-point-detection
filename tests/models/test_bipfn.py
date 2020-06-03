import torch
from app.models.bifpn import BiFPN


def test_bifpn() -> None:
    req = (torch.ones((1, 3, 10, 10)) for _ in range(5))
    m = BiFPN(channels=1)
    res = m(req)
    #  assert req.shape == res.shape
