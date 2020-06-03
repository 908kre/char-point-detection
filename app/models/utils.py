import numpy as np
import typing as t
from functools import partial
from torch import nn, Tensor


def round_filters(filters: t.Any, global_params: t.Any) -> t.Any:
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def normal_init(
    module: nn.Module, mean: float = 0, std: float = 1, bias: float = 0,
) -> None:
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias"):
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob: t.Any) -> t.Any:
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init
