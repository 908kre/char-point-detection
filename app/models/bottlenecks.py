import typing as t
from torch import nn, Tensor
import torch.nn.functional as F
from .modules import Hswish, ConvBR2d, CSE2d


class MobileV3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        kernel: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        padding = (kernel - 1) // 2
        self.stride = stride
        self.is_shortcut = (stride == 1) and (in_channels == out_channels)

        self.conv = nn.Sequential(
            # pw
            ConvBR2d(in_channels, mid_channels, 1, 1, 0),
            Hswish(),
            #  # dw
            ConvBR2d(
                mid_channels,
                mid_channels,
                kernel,
                stride,
                padding,
                groups=mid_channels,
                bias=False,
            ),
            CSE2d(mid_channels, reduction=4),
            Hswish(),
            # pw-linear
            ConvBR2d(mid_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):  # type: ignore
        if self.is_shortcut:
            if self.stride > 1:
                x = F.avg_pool2d(x, self.stride, self.stride)  # avg
            return x + self.conv(x)
        else:
            return self.conv(x)


class SENextBottleneck2d(nn.Module):
    pool: t.Union[None, nn.MaxPool2d, nn.AvgPool2d]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 8,
        groups: int = 16,
        pool: str = "max",
    ) -> None:
        super().__init__()
        mid_channels = groups * (out_channels // 2 // groups)
        self.down = nn.Sequential(
            ConvBR2d(in_channels, mid_channels, 1, 0, 1,),
            ConvBR2d(mid_channels, mid_channels, 3, 1, 1, groups=groups),
        )

        if stride > 1:
            if pool == "max":
                self.down.add_module("pool", nn.MaxPool2d(stride, stride))
            elif pool == "avg":
                self.down.add_module("pool", nn.AvgPool2d(stride, stride))

        self.conv = nn.Sequential(
            ConvBR2d(mid_channels, out_channels, 1, 0, 1,),
            Hswish(inplace=True),
            CSE2d(out_channels, reduction),
        )

        self.stride = stride
        self.is_shortcut = in_channels != out_channels
        self.activation = Hswish(inplace=True)
        if self.is_shortcut:
            self.shortcut = nn.Sequential(ConvBR2d(in_channels, out_channels, 1, 0, 1),)

    def forward(self, x: Tensor) -> Tensor:
        s = self.down(x)
        s = self.conv(s)
        if self.is_shortcut:
            if self.stride > 1:
                x = F.avg_pool2d(x, self.stride, self.stride)  # avg
            x = self.shortcut(x)
        x = x + s
        x = self.activation(x)
        return x
