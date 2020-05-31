import typing as t
from app.entities import Images
from torch.utils.data import Dataset as _Dataset


class Dataset(_Dataset):
    def __init__(self, images: Images) -> None:
        self.rows = list(images.values())

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> t.Any:
        image = self.rows[index]
        arr = image.get_arr()
        #  image = cv2.imread(f'{DATA_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        #  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        #  image /= 255.0
        #  if self.transforms:
        #      sample = {'image': image}
        #      sample = self.transforms(**sample)
        #      image = sample['image']
        #  return image, image_id
