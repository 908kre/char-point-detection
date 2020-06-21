import typing as t
from torch import Tensor

Image = t.NewType("Image", Tensor)
ImageBatch = t.NewType("ImageBatch", Tensor)
