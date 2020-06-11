from pathlib import Path
import json
import numpy as np
from torch.utils.data import Dataset
from .imagefile import ImageFileDataset


class NDLDataSet(Dataset):

    def __init__(self, image_dir, annot_dir, transforms=None):
        self.image_dataset = ImageFileDataset(image_dir, ".jpg")
        annot_files = sorted(Path(annot_dir).glob("*.json"))
        self.image_ids = [  # modern only
            _.stem for _ in annot_files if self._get_age(_) == 1
        ]
        self.transforms = transforms

    def __getitem__(self, i):
        image_id = self.image_ids[i]
        sample = self.image_dataset.lookup(image_id + ".jpg")
        sample["image_id"] = image_id
        sample["bboxes"] = np.zeros((0, 4), dtype=float)  # no boxes
        sample["labels"] = np.zeros(0, dtype=int)  # dummy
        if self.transforms is None:
            return sample
        else:
            return self.transforms(sample)

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def _get_age(annot_file):
        with open(annot_file) as fp:
            annot = json.load(fp)
            if annot["attributes"]["年代"] == "近代":
                return 1
            else:
                return 0
