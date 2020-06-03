import torch
from app.models.bottlenecks import MobileV3


def test_mobilev3():
    req = torch.ones((1, 32, 10, 10))
    m = MobileV3(in_channels=32, out_channels=16, mid_channels=32)
    res = m(req)
    assert res.shape == (1, 16, 10, 10)

    m = MobileV3(in_channels=32, out_channels=32, mid_channels=64)
    res = m(req)
    assert res.shape == (1, 32, 10, 10)
