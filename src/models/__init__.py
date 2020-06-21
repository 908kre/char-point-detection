import torch
from ..config import ModelConfig
from .efficientnet import EfficientNetBackbone
from .fusion import BiFPN
from .ctdet import CenterNetDetector


def get_model(config: ModelConfig) -> CenterNetDetector:
    backbone = EfficientNetBackbone(
        config.phi, config.pretrained and (config.weights == None)
    )
    depth = config.phi + 2
    width = round((1.35 ** config.phi) * 8) * 8
    fusion = BiFPN(backbone.sideout_channels[1:], width=width, depth=depth)
    model = CenterNetDetector(backbone, fusion)
    if config.weights is not None:
        state_dict = torch.load(
            config.weights, map_location=lambda storage, loc: storage
        )
        model.load_state_dict(state_dict)
    return model
