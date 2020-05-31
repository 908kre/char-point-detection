import numpy as np
import typing as t
from app.entities import Images
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, images: Images, mode:t.Literal["train", "test"]="train") -> None:
        self.rows = list(images.values())
        self.mode = mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> t.Any:
        image = self.rows[index]
        arr = image.get_arr()
        box_arrs = np.stack([x.to_arr() for x in image.bboxes])
        print(box_arrs.shape)
        print(arr.shape)
        print(arr.dtype)
        #  image = cv2.imread(f'{DATA_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        #  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        #  image /= 255.0
        #  if self.transforms:
        #      sample = {'image': image}
        #      sample = self.transforms(**sample)
        #      image = sample['image']
        #  return image, image_id
