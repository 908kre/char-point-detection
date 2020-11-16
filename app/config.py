from typing import Tuple
from typing_extensions import Literal
from dataclasses import dataclass
from object_detection.entities import PyramidIdx
from object_detection.models.centernetv1 import GaussianMapMode
from object_detection.models.backbones.effnet import Phi

# train
lr = 1e-4
T_max = 16
eta_min = 1e-6

# model
effdet_id: Phi = 1
channels = 128
pretrained = True
out_idx: PyramidIdx = 4
fpn_depth = 1
hm_depth = 1
box_depth = 1

# criterion
heatmap_weight = 1.0
box_weight = 20.0
sigma = 20.0
mode: GaussianMapMode = "aspect"

metric: Tuple[str, Literal["max", "min"]] = ("score", "max")
iou_thresholds = [0.5]

# to_boxes
confidence_threshold = 0.4
use_peak = True

seed = 777
device = "cuda"

max_size = 512 * 2
batch_size = 2
num_workers = 8

out_dir = f"/store/models/ctdtv1-effdet_id-{effdet_id}-fpn_depth-{fpn_depth}-hm_depth-{hm_depth}-box_depth-{box_depth}-channels-{channels}-out_idx-{out_idx}-max_size-{max_size}"
