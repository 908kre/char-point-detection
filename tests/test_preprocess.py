from app.preprocess import load_lables, plot_with_bbox
from app.entities import TrainLabel


def test_load_lables() -> None:
    load_lables()


def test_plot_with_bbox() -> None:
    label = TrainLabel(
        image_id="b6ab77fd7",
        width=1024,
        height=1024,
        bbox=(953, 220, 56, 103),
        source="usask_1",
    )
    plot_with_bbox(label)
