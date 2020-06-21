import typing as t
from torch import nn, Tensor
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):  # type:ignore
        return x * torch.sigmoid(x)


class Hswish(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x):  # type: ignore
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):  # type:ignore
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class CSE2d(nn.Module):
    def __init__(self, in_channels: int, reduction: int) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            Hsigmoid(inplace=True),
        )

    def forward(self, x):  # type: ignore
        x = x * self.se(x)
        return x


class ConvBR2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        activation: t.Optional[nn.Module] = Hswish(inplace=True),
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
