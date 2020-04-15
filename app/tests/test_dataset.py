import pytest
from app.dataset import Dataset, Mode
from app.entities import Annotations, Annotation
import typing as t
from skimage import io
from app.cache import Cache
from app.preprocess import load_labels, get_annotations
from torchvision.utils import save_image, make_grid

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
        #  ("0002fe0e341a9563d0c01b9dab820222", "Train",),
        ("0a0c21426ac8363577a4548348b76494", "Test",),
        ("0a0c21426ac8363577a4548348b76494", "Train",),
        #  ("005181579e9e1b7bcb42956b4cfbdba8", "Test",),
        #  ("0094c096b31a1bace6449743e78b861b", "Test",),
    ],
)
def test_transform(id: str, mode: Mode) -> None:
    annotations = [Annotation(id, [])]

    dataset = Dataset(annotations, mode=mode, resolution=128)
    save_image(
        make_grid(
            [dataset[0][0] for i in range(24)],
            nrow=8,
            padding=2,
            normalize=False,
            range=None,
            scale_each=False,
            pad_value=0,
        ),
        f"/store/tmp/test_aug_{id}-{mode}.png",
    )
