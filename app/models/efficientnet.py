import typing as t
import torch
from torch import nn
from torch.nn import functional as F

from .utils import round_filters


#  class EfficientNet(nn.Module):
#      """
#      An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
#      Args:
#          blocks_args (list): A list of BlockArgs to construct blocks
#          global_params (namedtuple): A set of GlobalParams shared between blocks
#      Example:
#          model = EfficientNet.from_pretrained('efficientnet-b0')
#      """
#
#      def __init__(self, blocks_args: t.Any = None, global_params: t.Any = None) -> None:
#          super().__init__()
#          assert isinstance(blocks_args, list), "blocks_args should be a list"
#          assert len(blocks_args) > 0, "block args must be greater than 0"
#          self._global_params = global_params
#          self._blocks_args = blocks_args
#
#          # Get static or dynamic convolution depending on image size
#          Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
#
#          # Batch norm parameters
#          bn_mom = 1 - self._global_params.batch_norm_momentum
#          bn_eps = self._global_params.batch_norm_epsilon
#
#          # Stem
#          in_channels = 3  # rgb
#          # number of output channels
#          out_channels = round_filters(32, self._global_params)
#          self._conv_stem = Conv2d(
#              in_channels, out_channels, kernel_size=3, stride=2, bias=False
#          )
#          self._bn0 = nn.BatchNorm2d(
#              num_features=out_channels, momentum=bn_mom, eps=bn_eps
#          )
#
#          # Build blocks
#          self._blocks = nn.ModuleList([])
#          for i in range(len(self._blocks_args)):
#              # Update block input and output filters based on depth multiplier.
#              self._blocks_args[i] = self._blocks_args[i]._replace(
#                  input_filters=round_filters(
#                      self._blocks_args[i].input_filters, self._global_params
#                  ),
#                  output_filters=round_filters(
#                      self._blocks_args[i].output_filters, self._global_params
#                  ),
#                  num_repeat=round_repeats(
#                      self._blocks_args[i].num_repeat, self._global_params
#                  ),
#              )
#
#              # The first block needs to take care of stride and filter size increase.
#              self._blocks.append(MBConvBlock(self._blocks_args[i], self._global_params))
#              if self._blocks_args[i].num_repeat > 1:
#                  self._blocks_args[i] = self._blocks_args[i]._replace(
#                      input_filters=self._blocks_args[i].output_filters, stride=1
#                  )
#              for _ in range(self._blocks_args[i].num_repeat - 1):
#                  self._blocks.append(
#                      MBConvBlock(self._blocks_args[i], self._global_params)
#                  )
#
#          # Head'efficientdet-d0': 'efficientnet-b0',
#          # output of final block
#          in_channels = self._blocks_args[len(self._blocks_args) - 1].output_filters
#          out_channels = round_filters(1280, self._global_params)
#          self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#          self._bn1 = nn.BatchNorm2d(
#              num_features=out_channels, momentum=bn_mom, eps=bn_eps
#          )
#
#          # Final linear layer
#          self._avg_pooling = nn.AdaptiveAvgPool2d(1)
#          self._dropout = nn.Dropout(self._global_params.dropout_rate)
#          self._fc = nn.Linear(out_channels, self._global_params.num_classes)
#          self._swish = MemoryEfficientSwish()
#
#      def set_swish(self, memory_efficient: bool = True) -> None:
#          """Sets swish function as memory efficient (for training) or standard (for export)"""
#          self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
#          for block in self._blocks:
#              block.set_swish(memory_efficient)
#
#      def extract_features(self, inputs: t.List[t.Any]) -> None:
#          """ Returns output of the final convolution layer """
#          # Stem
#          x = self._swish(self._bn0(self._conv_stem(inputs)))
#
#          P = []
#          index = 0
#          num_repeat = 0
#          # Blocks
#          for idx, block in enumerate(self._blocks):
#              drop_connect_rate = self._global_params.drop_connect_rate
#              if drop_connect_rate:
#                  drop_connect_rate *= float(idx) / len(self._blocks)
#              x = block(x, drop_connect_rate=drop_connect_rate)
#              num_repeat = num_repeat + 1
#              if num_repeat == self._blocks_args[index].num_repeat:
#                  num_repeat = 0
#                  index = index + 1
#                  P.append(x)
#          return P
#
#      def forward(self, inputs):  # type: ignore
#          """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
#          # Convolution layers
#          P = self.extract_features(inputs)
#          return P
#
#      @classmethod
#      def from_pretrained(
#          cls, model_name: str, num_classes: int = 1000, in_channels: int = 3
#      ) -> EfficientNet:
#          model = cls.from_name(model_name, override_params={"num_classes": num_classes})
#          load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
#          if in_channels != 3:
#              Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
#              out_channels = round_filters(32, model._global_params)
#              model._conv_stem = Conv2d(
#                  in_channels, out_channels, kernel_size=3, stride=2, bias=False
#              )
#          return model
#
#      @classmethod
#      def from_pretrained(cls, model_name: str, num_classes: int = 1000) -> EfficientNet:
#          model = cls.from_name(model_name, override_params={"num_classes": num_classes})
#          load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
#
#          return model
#
#      @classmethod
#      def get_image_size(cls, model_name: str) -> t.Any:
#          cls._check_model_name_is_valid(model_name)
#          _, _, res, _ = efficientnet_params(model_name)
#          return res
#
#      @classmethod
#      def _check_model_name_is_valid(
#          cls, model_name: str, also_need_pretrained_weights: bool = False
#      ) -> None:
#          """ Validates model name. None that pretrained weights are only available for
#          the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
#          num_models = 4 if also_need_pretrained_weights else 8
#          valid_models = ["efficientnet-b" + str(i) for i in range(num_models)]
#          if model_name not in valid_models:
#              raise ValueError("model_name should be one of: " + ", ".join(valid_models))
#
#      def get_list_features(self) -> t.List[t.Any]:
#          list_feature = []
#          for idx in range(len(self._blocks_args)):
#              list_feature.append(self._blocks_args[idx].output_filters)
#
#          return list_feature
