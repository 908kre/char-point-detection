import pytest
from app.dataset import Dataset, CDataset
import typing as t
from app.cache import Cache


#  @pytest.mark.parametrize("window_size, expected", [(10, 3), (15, 2), (5, 4)])
#  def test_dataset(window_size: int, expected: int) -> None:
#      cache = Cache("/store/tmp")
#      df = cache("load-train", load)("/store/data/train.csv")
#      d = Dataset(df[:20], window_size=window_size, stride=5,)
#      assert len(d) == expected
#      x, y = d[0]
#      assert x.shape == (1, window_size)
#      assert y.shape == (window_size,)
#
#
#  def test_dataset_total_size() -> None:
#      cache = Cache("/store/tmp")
#      df = cache("load-train", load)("/store/data/train.csv")[:20]
#      d = Dataset(df, window_size=5, stride=5,)
#      total_size = 0
#      for i in range(len(d)):
#          t, _ = d[i]
#          total_size += t.shape[1]
#      assert total_size == 20
#
#
#  def test_c_dataset_total_size() -> None:
#      cache = Cache("/store/tmp")
#      df = cache("load-train", load)("/store/data/train.csv")[:20]
#      d = CDataset(df, window_size=5, stride=5,)
#      total_size = 0
#      for i in range(len(d)):
#          x, y = d[i]
#          assert y.shape == (1,)
#          total_size += x.shape[1]
#      assert total_size == 20
