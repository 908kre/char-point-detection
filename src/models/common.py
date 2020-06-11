import torch.nn as nn


def init_weights(m: nn.Module, recursive=False):
    if hasattr(m, "_init_weights"):
        pass
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif recursive:
        for m in m.children():
            init_weights(m, recursive)
