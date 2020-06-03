import torch
import typing as t
from torch import nn, Tensor
import torch.nn.functional as F
from .bottlenecks import MobileV3


class Down2d(nn.Module):
    def __init__(
        self, channels: int, bilinear: bool = False, merge: bool = True,
    ) -> None:
        super().__init__()
        self.down = MobileV3(
            in_channels=channels,
            out_channels=channels,
            mid_channels=channels // 2,
            stride=2,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.down(x)
        return x


class Up2d(nn.Module):
    up: t.Union[nn.Upsample, nn.ConvTranspose2d]

    def __init__(
        self, channels: int, bilinear: bool = False, merge: bool = True,
    ) -> None:
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.merge = merge
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)

    def forward(self, x, t):  # type: ignore
        x = self.up(x)
        diff_h = torch.tensor([x.size()[2] - t.size()[2]])
        diff_w = torch.tensor([x.size()[3] - t.size()[3]])
        x = F.pad(x, (diff_h - diff_h // 2, diff_w - diff_w // 2))
        return x


class Merge2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,) -> None:
        super().__init__()
        self.merge = MobileV3(
            in_channels=in_channels,
            out_channels=out_channels,
            mid_channels=out_channels // 2,
            stride=1,
        )

    def forward(self, *inputs: Tensor) -> Tensor:
        x = torch.cat(inputs, dim=1)
        x = self.merge(x)
        return x


FP = t.Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]  # p3, p4, p5, p6, p7


class BiFPN(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.up3 = Up2d(channels)
        self.up4 = Up2d(channels)
        self.up5 = Up2d(channels)
        self.up6 = Up2d(channels)
        self.up7 = Up2d(channels)

        self.mid6 = Merge2d(in_channels=channels * 2, out_channels=channels)
        self.mid5 = Merge2d(in_channels=channels * 2, out_channels=channels)
        self.mid4 = Merge2d(in_channels=channels * 2, out_channels=channels)

        self.out3 = Merge2d(in_channels=channels * 2, out_channels=channels)
        self.out4 = Merge2d(in_channels=channels * 3, out_channels=channels)
        self.out5 = Merge2d(in_channels=channels * 3, out_channels=channels)
        self.out6 = Merge2d(in_channels=channels * 3, out_channels=channels)
        self.out7 = Merge2d(in_channels=channels * 2, out_channels=channels)

        self.down3 = Down2d(channels)
        self.down4 = Down2d(channels)
        self.down5 = Down2d(channels)
        self.down6 = Down2d(channels)
        self.down7 = Down2d(channels)

    def forward(self, inputs: FP) -> FP:
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        p7_up = self.up7(p7_in, p6_in)
        p6_mid = self.mid6(p7_up, p6_in)
        p6_up = self.up7(p6_mid, p5_in)
        p5_mid = self.mid5(p6_up, p5_in)
        p5_up = self.up5(p5_mid, p4_in)
        p4_mid = self.mid4(p5_up, p4_in)
        p4_up = self.up4(p4_mid, p3_in)

        p3_out = self.out3(p3_in, p4_up)
        p3_down = self.down3(p3_out)
        p4_out = self.out4(p3_down, p4_mid, p4_in)
        p4_down = self.down4(p4_out)
        p5_out = self.out5(p4_down, p5_mid, p5_in)
        p5_down = self.down5(p5_out)
        p6_out = self.out6(p5_down, p6_mid, p6_in)
        p6_down = self.down6(p6_out)
        p7_out = self.out7(p6_down, p7_in)

        return p3_out, p4_out, p5_out, p6_out, p7_out
