from torch.utils.data import Dataset
from .kuzushiji import KuzushijiDataset
from .ndl import NDLDataSet
from .coco import CocoDataset

NORMALIZE_PARAMS = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

DATASETS = dict(kuzushiji=KuzushijiDataset, ndl=NDLDataSet, coco=CocoDataset)


def get_dataset(name: str, image_source: str, annot_source: str) -> Dataset:
    return DATASETS[name](image_source, annot_source)
