import typing as t
from torch import nn
import torch.nn.functional as F
from .modules import Hswish, ConvBR2d, CSE2d


class MobileV3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: t.Literal[3, 4, 5],
        stride: t.Literal[1, 2],
        mid_channels: int,
    ):
        super().__init__()
        padding = (kernel - 1) // 2
        self.is_short_cut = stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            # pw
            ConvBR2d(in_channels, mid_channels, 1, 1, 0),
            Hswish(),
            #  # dw
            ConvBR2d(mid_channels, mid_channels, 1, 1, 0),
            CSE2d(mid_channels, reduction=4),
            Hswish(),
            # pw-linear
            ConvBR2d(mid_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):  # type: ignore
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
