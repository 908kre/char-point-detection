import torch
import numpy as np
from cytoolz.curried import groupby, valmap, pipe, unique, map, reduce
from pathlib import Path
import typing as t
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from .data.coco import CocoDataset
from torch.utils.data import DataLoader, Subset, ConcatDataset
from object_detection.models.backbones.effnet import EfficientNetBackbone
from object_detection.metrics.mean_precision import MeanPrecition
from object_detection.models.centernetv1 import (
    collate_fn,
    CenterNetV1,
    Trainer,
    Visualize,
    Reg,
    ToBoxes,
)
from object_detection.model_loader import ModelLoader, BestWatcher
from app import config
from app.preprocess import kfold


def train() -> None:
    val_dataset = CocoDataset(
        image_dir="/store/datasets/preview",
        annot_file="/store/datasets/preview/20200611_coco_imglab.json",
        max_size=config.max_size,
    )
    train_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        val_dataset,
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
    visualize = Visualize("./", "centernet", limit=10, use_alpha=True)
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
