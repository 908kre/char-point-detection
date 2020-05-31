from app.dataset import TrainDataset
from app.preprocess import load_lables


def test_dataset() -> None:
    images = load_lables(limit=10)
    dataset = TrainDataset(images)
    item = dataset[0]
    print(item)
