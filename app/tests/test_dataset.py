import pytest
from app.dataset import Dataset
import typing as t
from app.cache import Cache
from app.pipeline import load


@pytest.mark.parametrize("window_size, expected", [(10, 3), (15, 2), (5, 4)])
def test_dataset(window_size: int, expected: int) -> None:
    cache = Cache("/store/tmp")
    df = cache("load-train", load)("/store/data/train.csv")
    d = Dataset(df[:20], window_size=window_size, stride=5,)
    assert len(d) == expected
    x, y = d[0]
    assert x.shape == (1, window_size)
    assert y.shape == (1, window_size)
