import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import init_weights


def cbr(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def ctbr(in_channels, out_channels, kernel_size=2, stride=1, bias=False):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class BiFPN(nn.Module):
    def __init__(self, in_channels=[64, 64, 128, 256, 512], width=64, depth=2):
        super().__init__()
        self.out_channels = width
        self.in_conv = nn.ModuleList(
            [cbr(_, width, kernel_size=1) for _ in in_channels]
        )
        self.bifpn = nn.Sequential(
            *[BiFPNUnit([width] * len(in_channels)) for _ in range(depth)]
        )
        self.prefuse_conv = nn.ModuleList([cbr(width, width) for _ in in_channels])
        self.fusion = FastNormalizedFusion(len(in_channels))
        self.out_conv = cbr(width, width)

    def _init_weights(self):
        init_weights(self.in_conv, recursive=True)
        init_weights(self.prefuse_conv, recursive=True)
        init_weights(self.out_conv, recursive=True)

    def forward(self, x):
        x = [conv(t) for conv, t in zip(self.in_conv, x)]
        x = self.bifpn(x)
        x = self.fusion(
            *[
                conv(F.interpolate(t, size=x[0].shape[2:]))
                for conv, t in zip(self.prefuse_conv, x)
            ]
        )
        return self.out_conv(x)


class BiFPNUnit(nn.Module):
    def __init__(self, in_channels=[64, 64, 128, 256, 512]):
        super().__init__()
        depth = len(in_channels)
        self.upsample = nn.ModuleList(
            [
                ctbr(in_channels[depth - 1 - i], in_channels[depth - 2 - i], stride=2)
                for i in range(depth - 1)
            ]
        )
        self.topdown_fusion = nn.ModuleList(
            [FastNormalizedFusion(2) for i in range(depth - 1)]
        )
        self.topdown_conv = nn.ModuleList(
            [
                cbr(in_channels[depth - 2 - i], in_channels[depth - 2 - i])
                for i in range(depth - 1)
            ]
        )
        self.downsample = nn.ModuleList(
            [
                cbr(in_channels[i], in_channels[i + 1], stride=2)
                for i in range(depth - 1)
            ]
        )
        self.bottomup_fusion = nn.ModuleList(
            [
                FastNormalizedFusion(3 if i < (depth - 2) else 2)
                for i in range(depth - 1)
            ]
        )
        self.bottomup_conv = nn.ModuleList(
            [cbr(in_channels[i + 1], in_channels[i + 1]) for i in range(depth - 1)]
        )
        self._init_weights()

    def _init_weights(self):
        init_weights(self.upsample, recursive=True)
        init_weights(self.topdown_conv, recursive=True)
        init_weights(self.downsample, recursive=True)
        init_weights(self.bottomup_conv, recursive=True)

    def forward(self, x):
        assert isinstance(x, list)
        depth = len(x)

        # top-down
        td = []
        for i in range(len(x)):
            t = x[-i - 1]
            if i > 0:
                t = self.topdown_fusion[i - 1](t, self.upsample[i - 1](td[-1]))
                t = self.topdown_conv[i - 1](t)
            td.append(t)

        # bottom-up
        bu = []
        for i in range(len(x)):
            t = td[depth - 1 - i]
            if i > 0:
                feats = [t]
                feats.append(self.downsample[i - 1](bu[-1]))
                if i < (depth - 1):
                    feats.append(x[i])
                t = self.bottomup_fusion[i - 1](*feats)
                t = self.bottomup_conv[i - 1](t)
            bu.append(t)

        return bu


class FastNormalizedFusion(nn.Module):
    def __init__(self, num_features, eps=1e-7):
        super().__init__()
        self.num_features = num_features
        self.weights = torch.nn.Parameter(torch.ones(num_features))
        self.eps = eps

    def forward(self, *x):
        assert self.num_features == len(x)
        weights = self.weights / (self.weights.mean() + self.eps)
        return torch.stack(x).mul_(weights[:, None, None, None, None]).sum(dim=0)
