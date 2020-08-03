import torch
import numpy as np
from cytoolz.curried import groupby, valmap, pipe, unique, map, reduce
from pathlib import Path
from typing import Any, List, Tuple
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from object_detection.models.backbones.effnet import EfficientNetBackbone
from object_detection.metrics.mean_precision import MeanPrecition
from object_detection.models.centernetv1 import (
    CenterNetV1,
    Trainer as _Trainer,
    Visualize,
    Reg,
    ToBoxes,
)
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from object_detection.entities import (
    TrainSample,
    ImageBatch,
    YoloBoxes,
    ImageId,
    Labels,
)
from object_detection.model_loader import ModelLoader, BestWatcher
from app import config
from torch.utils.data import DataLoader, Subset, ConcatDataset
from .data.coco import CocoDataset
from .data.kuzushiji import CodhKuzushijiDataset
from .data.negative import NegativeDataset


def collate_fn(
    batch: List[TrainSample],
) -> Tuple[ImageBatch, List[YoloBoxes], List[Labels], List[ImageId]]:
    images: List[Any] = []
    id_batch: List[Any] = []
    box_batch: List[Any] = []
    label_batch: List[Any] = []

    for id, img, boxes, labels in batch:
        images.append(img)
        box_batch.append(boxes)
        id_batch.append(id)
        label_batch.append(labels)
    return ImageBatch(torch.stack(images)), box_batch, label_batch, id_batch

class Trainer(_Trainer):
    def __init__(self, *args:Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lr_scheduler = CosineAnnealingLR(
            optimizer=self.optimizer, T_max=config.T_max, eta_min=config.eta_min
        )
        self.scaler = GradScaler()

    def train_one_epoch(self) -> None:
        self.model.train()
        loader = self.train_loader
        for images, box_batch, ids, _ in tqdm(loader):
            self.optimizer.zero_grad()
            images, box_batch = self.preprocess((images, box_batch))
            outputs = self.model(images)
            loss, hm_loss, box_loss, gt_hms = self.criterion(images, outputs, box_batch)
            self.scaler.scale(loss).backward()
            self.optimizer.step()
            self.meters["train_loss"].update(loss.item())
            self.meters["train_box"].update(box_loss.item())
            self.meters["train_hm"].update(hm_loss.item())
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
        ConcatDataset([codh, neg]),
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
