import typing as t
from torch import Tensor
CoCoBoxes = t.NewType("CoCoBoxes", Tensor)
Labels = t.NewType("Labels", Tensor)
