import torch
from app.models.modules import (
    Hswish,
    Hsigmoid,
    CSE2d,
)


def test_hswish() -> None:
    req = torch.zeros((1,))
    m = Hswish()
    res = m(req)
    print(res)


def test_hsigmoid() -> None:
    req = torch.zeros((1,))
    m = Hsigmoid()
    res = m(req)


def test_cse2d() -> None:
    req = torch.ones((1, 3, 10, 10))
    m = CSE2d(3, 2)
    res = m(req)
    assert req.shape == res.shape
