import typing as t
from torch import Tensor

ImageId = t.NewType("ImageId", str)
Image = t.NewType("Image", Tensor)  # [C, H, W]
ImageBatch = t.NewType("ImageBatch", Tensor)  # [B, C, H, W]
