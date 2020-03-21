import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSEModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv1d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):  # type: ignore
        return x * self.cSE(x) + x * self.sSE(x)


class SENextBottleneck(nn.Module):
    pool: t.Union[None, nn.MaxPool1d, nn.AvgPool1d]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 32,
        reduction: int = 16,
        pool: t.Optional[t.Literal["max", "avg"]] = None,
        is_shortcut: bool = False,
    ) -> None:
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvBR1d(in_channels, mid_channels, 1, 0, 1,)
        self.conv2 = ConvBR1d(mid_channels, mid_channels, 3, 1, 1, groups=groups)
        self.conv3 = ConvBR1d(mid_channels, out_channels, 1, 0, 1, is_activation=False)
        self.se = SEModule(out_channels, reduction)
        self.stride = stride
        self.is_shortcut = is_shortcut

        if is_shortcut:
            self.shortcut = ConvBR1d(
                in_channels, out_channels, 1, 0, 1, is_activation=False
            )
        if stride > 1:
            if pool == "max":
                self.pool = nn.MaxPool1d(stride, stride)
            elif pool == "avg":
                self.pool = nn.AvgPool1d(stride, stride)

    def forward(self, x):  # type: ignore
        s = self.conv1(x)
        s = self.conv2(s)
        if self.stride > 1:
            s = self.pool(s)
        s = self.conv3(s)
        s = self.se(s)

        if self.is_shortcut:
            if self.stride > 1:
                x = F.avg_pool1d(x, self.stride, self.stride)  # avg
            x = self.shortcut(x)

        x = x + s
        x = F.relu(x, inplace=True)

        return x


class ConvBR1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
        is_activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.is_activation = is_activation

        if is_activation:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # type: ignore
        x = self.bn(self.conv(x))
        if self.is_activation:
            x = self.relu(x)
        return x


class SEModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, in_channels // reduction, kernel_size=1, padding=0
        )
        self.conv2 = nn.Conv1d(
            in_channels // reduction, in_channels, kernel_size=1, padding=0
        )

    def forward(self, x):  # type: ignore
        # x: [B, C, H]
        s = F.adaptive_avg_pool1d(x, 1)  # [B, C, 1]
        s = self.conv1(s)  # [B, C//reduction, 1]
        s = F.relu(s, inplace=True)
        s = self.conv2(s)  # [B, C, 1]
        x = x + torch.sigmoid(s)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int,) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # type: ignore
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int,) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):  # type: ignore
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    up: t.Union[nn.Upsample, nn.ConvTranspose1d]

    def __init__(
        self, in_channels: int, out_channels: int, bilinear: bool = True
    ) -> None:
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):  # type: ignore
        x1 = self.up(x1)
        diff = torch.tensor([x2.size()[2] - x1.size()[2]])
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):  # type: ignore
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, bilinear: bool = True) -> None:
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear
        self.n_classes = n_classes

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):  # type: ignore
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class CNet(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = 1
        self.out_channels = 1

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # type: ignore
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.out(x5).view(-1, 1)
        return x
