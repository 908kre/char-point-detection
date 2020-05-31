from app.dataset import Dataset
from app.preprocess import load_lables


def test_dataset() -> None:
    images = load_lables(limit=10)
    dataset = Dataset(images)
    item = dataset[0]
