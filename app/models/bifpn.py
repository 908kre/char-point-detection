import typing as t
from torch import nn, Tensor
from .bottlenecks import MobileV3


class Down2d(nn.Module):
    """Upscaling to target image"""

    def __init__(
        self, channels: int, bilinear: bool = False, merge: bool = True,
    ) -> None:
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
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
    """Upscaling to target image"""

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


FP = t.Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]  # p3, p4, p5, p6, p7


class BiFPN(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

    def forward(self, inputs: FP) -> FP:
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs
        print(p3_in)

        return p3_in, p4_in, p5_in, p6_in, p7_in
