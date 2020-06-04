import torch
import numpy as np
import typing as t
from torch import nn, Tensor


class Anchors(nn.Module):
    def __init__(
        self,
        pyramid_levels: t.List[int] = [3, 4, 5, 6, 7],
        ratios: t.List[float] = [0.5, 1, 2],
        scales: t.List[float] = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
    ) -> None:
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.strides = [2 ** x for x in self.pyramid_levels]
        self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        self.ratios = ratios
        self.scales = scales

    def forward(self, image: Tensor) -> Tensor:
        """
        image: [B, C, W, H]
        return: [B, num_anchors, 4]
        """
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        feature_shapes = [
            (image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels
        ]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for feature_shape, size, stride in zip(
            feature_shapes, self.sizes, self.strides,
        ):
            anchors = generate_anchors(
                base_size=size, ratios=self.ratios, scales=self.scales
            )
            shifted_anchors = shift(feature_shape, stride, anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        return torch.from_numpy(all_anchors.astype(np.float32)).to(image.device)


def shift(shape: t.Any, stride: int, anchors: t.Any) -> t.Any:
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack(
        (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())
    ).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose(
        (1, 0, 2)
    )
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(
    base_size: int, ratios: t.List[float], scales: t.List[float],
) -> t.Any:
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors
