import pytest
from app.dataset import Dataset
from app.entities import Annotations
import typing as t
from app.cache import Cache
from app.preprocess import load_labels, get_annotations

cache = Cache("/store/tmp")


def test_dataset() -> None:
    labels = cache("labels", load_labels)("/store/dataset/labels.csv")
    annotations = cache("train_annotations", get_annotations)(
        "/store/dataset/train.csv", labels
    )

    d = Dataset(annotations)
    assert len(d) == 142119
    print(d[0])


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
