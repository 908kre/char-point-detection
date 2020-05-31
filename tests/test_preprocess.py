import typing as t
from app.entities import Image
from app.preprocess import load_lables, plot_with_bbox, kfold

first_sample: t.Union[None, Image] = None


def test_load_lables() -> None:
    images = load_lables()
    first_sample = next(iter(images.values()))
    assert len(first_sample.bboxs) == 47


def test_plot_with_bbox() -> None:
    images = load_lables()
    first_sample = next(iter(images.values()))
    plot_with_bbox(first_sample)


def test_kfold() -> None:
    images = load_lables()
    first_fold = next(kfold(images))
    print(first_fold)
