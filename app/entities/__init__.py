import typing as t
from .box import CoCoBoxes, Labels, PredBoxes, LabelBoxes, Confidences, YoloBoxes
from .image import Image, ImageBatch, ImageId

Sample = t.Tuple[ImageId, Image, YoloBoxes]
Batch = t.List[Sample]
