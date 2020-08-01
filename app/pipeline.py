import torch
import numpy as np
from cytoolz.curried import groupby, valmap, pipe, unique, map, reduce
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from object_detection.models.backbones.effnet import EfficientNetBackbone
from object_detection.metrics.mean_precision import MeanPrecition
from object_detection.models.centernetv1 import (
    collate_fn,
    CenterNetV1,
    Trainer as _Trainer,
    Visualize,
    Reg,
    ToBoxes,
)
from object_detection.model_loader import ModelLoader, BestWatcher
from app import config
from torch.utils.data import DataLoader, Subset, ConcatDataset
from .data.coco import CocoDataset
from .data.kuzushiji import CodhKuzushijiDataset
from .data.negative import NegativeDataset


class Trainer(_Trainer):
    def __init__(self, *args:Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lr_scheduler = CosineAnnealingLR(
            optimizer=self.optimizer, T_max=config.T_max, eta_min=config.eta_min
        )

    def train_one_epoch(self) -> None:
        super().train_one_epoch()
        self.lr_scheduler.step()


def train() -> None:
    coco = CocoDataset(
        image_dir="/store/datasets/preview",
        annot_file="/store/datasets/preview/20200611_coco_imglab.json",
        max_size=config.max_size,
    )
    codh = CodhKuzushijiDataset(
        image_dir="/store/codh-kuzushiji/resized",
        annot_file="/store/codh-kuzushiji/resized/annot.json",
    )
    neg = NegativeDataset(image_dir="/store/negative/images",)
    train_loader = DataLoader(
        ConcatDataset([codh, neg]), # type: ignore
        batch_size=config.batch_size,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        coco,
        batch_size=config.batch_size,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    backbone = EfficientNetBackbone(
        config.effdet_id, out_channels=config.channels, pretrained=config.pretrained
    )
    model = CenterNetV1(
        channels=config.channels,
        backbone=backbone,
        out_idx=config.out_idx,
        fpn_depth=config.fpn_depth,
        hm_depth=config.hm_depth,
        box_depth=config.box_depth,
    )
    model_loader = ModelLoader(
        out_dir=config.out_dir,
        key=config.metric[0],
        best_watcher=BestWatcher(mode=config.metric[1]),
    )
    visualize = Visualize(config.out_dir, "centernet", limit=10, use_alpha=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,)
    get_score = MeanPrecition(iou_thresholds=config.iou_thresholds)
    to_boxes = ToBoxes(threshold=config.confidence_threshold, use_peak=config.use_peak,)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        model_loader=model_loader,
        optimizer=optimizer,
        visualize=visualize,
        device=config.device,
        get_score=get_score,
        to_boxes=to_boxes,
    )
    trainer(1000)


def submit() -> None:
    ...
