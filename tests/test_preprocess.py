import typing as t
from app.entities import Image
from app.preprocess import load_lables, plot_with_bbox, KFold

first_sample: t.Union[None, Image] = None


def test_load_lables() -> None:
    images = load_lables()
    first_sample = next(iter(images.values()))
    assert len(first_sample.bboxes) == 47


def test_plot_with_bbox() -> None:
    images = load_lables()
    first_sample = next(iter(images.values()))
    plot_with_bbox(first_sample)


def test_kfold() -> None:
    images = load_lables()
    kf = KFold()
    fold_train, fold_valid = next(kf(images))
    assert len(fold_train) + len(fold_valid) == len(images)
