import typing as t
from torch import Tensor

Image = t.NewType("Image", Tensor)  # [C, H, W]
ImageBatch = t.NewType("ImageBatch", Tensor)  # [B, C, H, W]
