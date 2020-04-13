import pytest
from app.dataset import Dataset, Mode
from app.entities import Annotations, Annotation
import typing as t
from skimage import io
from app.cache import Cache
from app.preprocess import load_labels, get_annotations
from torchvision.utils import save_image

cache = Cache("/store/tmp")


def test_dataset() -> None:
    labels = cache("labels", load_labels)("/store/dataset/labels.csv")
    annotations = cache("train_annotations", get_annotations)(
        "/store/dataset/train.csv", labels
    )
    d = Dataset(annotations)
    assert len(d) == 142119


@pytest.mark.parametrize(
    "id, mode",
    [
        ("0002fe0e341a9563d0c01b9dab820222", "Train",),
        ("0a0c21426ac8363577a4548348b76494", "Test",),
    ],
)
def test_transform(id: str, mode: Mode) -> None:
    annotations = [Annotation(id, [])]

    dataset = Dataset(annotations, mode=mode,resolution=128)
    for i in range(10):
        img, _ = dataset[0]
        save_image(img, f"/store/tmp/aug_{id}_{i}.png")
