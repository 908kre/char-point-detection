import typing as t
from torch.utils.data import Dataset as _Dataset


class Dataset(_Dataset):
    def __init__(
        self,
        df: t.Any,
        window_size: int,
        stride: int,
        mode: t.Literal["train", "test"] = "train",
    ) -> None:
        self.df = df
        self.mode = mode
        self.cache: t.Dict[int, t.Any] = {}
        self.window_size = window_size
        self.stride = stride

    def __len__(self) -> int:
        return (len(self.df) - self.window_size) // self.stride + 1

    def __getitem__(self, idx: int) -> t.Tuple[t.Any, t.Any]:
        row = self.df.iloc[idx * self.stride : idx * self.stride + self.window_size]
        if idx in self.cache:
            return self.cache[idx]
        if self.mode == "train":
            res = (
                row[["signal"]].values.transpose(),
                row[["open_channels"]].values.transpose(),
            )
        else:
            res = (row[["signal"]].values.transpose(), None)
        self.cache[idx] = res
        return res
