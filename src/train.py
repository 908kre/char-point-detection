import random
from datetime import datetime as dt
from pathlib import Path
from collections import defaultdict
from functools import partial
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Lambda
from radam import RAdam
import cv2
import albumentations as albm
import albumentations.pytorch as albm_torch
from tqdm import tqdm
from .losses.focal_loss import FocalLossWithLogits
from .config import (
    TrainConfig, ModelConfig, DataConfig, DatasetConfig,
    OptimizerConfig, SchedulerConfig, ValidConfig
)
from .models import get_model
from .models.ctdet import get_bboxes
from .model_saver import NullSaver, BestModelSaver
from .meters import MeanMeter, EMAMeter
from .data import get_dataset, NORMALIZE_PARAMS
from .data.common import Transformed, collate_fn
from .transforms import RandomDilateErode, MakeMap
from .evaluation import coco_to_pascal, sweep_average_precision


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_loaders(config: DataConfig):

    train_dataset = ConcatDataset([
        get_dataset(c.name, c.image, c.annot)
        for c in config.train_datasets
    ])

    val_dataset = ConcatDataset([
        get_dataset(c.name, c.image, c.annot)
        for c in config.val_datasets
    ])

    train_transforms = albm.Compose([
        albm.LongestMaxSize(config.input_size),
        albm.PadIfNeeded(
            config.input_size, config.input_size,
            border_mode=cv2.BORDER_CONSTANT, value=0),
        albm.VerticalFlip(),
        albm.RandomRotate90(),
        MakeMap(config.hm_alpha),
        albm.Normalize(**NORMALIZE_PARAMS),
        albm_torch.ToTensorV2()
    ], bbox_params={"format": "coco", "label_fields": ["labels"]})

    val_transforms = albm.Compose([
        albm.LongestMaxSize(config.input_size),
        albm.PadIfNeeded(
            config.input_size, config.input_size,
            border_mode=cv2.BORDER_CONSTANT, value=0),
        MakeMap(config.hm_alpha),
        albm.Normalize(**NORMALIZE_PARAMS),
        albm_torch.ToTensorV2()
    ], bbox_params={"format": "coco", "label_fields": ["labels"]})

    train_dataset = Transformed(
        train_dataset,
        transforms=Lambda(lambda _: train_transforms(**_)))

    val_dataset = Transformed(
        val_dataset,
        transforms=Lambda(lambda _: val_transforms(**_)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size, drop_last=True, shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=lambda worker_id: init_seed(config.seed + worker_id),
        num_workers=10, pin_memory=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size, drop_last=False, shuffle=False,
        collate_fn=collate_fn,
        num_workers=10, pin_memory=True)

    return train_loader, val_loader


def get_optimizer(config: OptimizerConfig, parameters):
    optimizers = dict(
        radam=RAdam
    )
    if config.name in optimizers:
        return optimizers[config.name](parameters, **config.config)
    else:
        raise NotImplementedError


def get_scheduler(config: SchedulerConfig, optimizer):
    schedulers = dict(
        cosine=CosineAnnealingLR,
        step=MultiStepLR
    )
    if config.name in schedulers:
        return schedulers[config.name](optimizer, **config.config)
    else:
        raise NotImplementedError


def get_criterions(config: dict):
    criterions = {}
    for key, (loss_config, weight) in config.items():
        if key == "hm":
            criterion = FocalLossWithLogits(**loss_config)
        else:
            criterion = nn.L1Loss(reduction="none", **loss_config)
        criterions[key] = (criterion, weight)
    return criterions


def train(config: TrainConfig):

    init_seed(config.seed)

    device = torch.device(config.device)

    model = get_model(config.model)
    dp_model = nn.DataParallel(model)

    train_loader, val_loader = get_loaders(config.data)

    optimizer = get_optimizer(config.optimizer, model.parameters())
    lr_scheduler = get_scheduler(config.scheduler, optimizer)

    criterions = get_criterions(config.loss)

    if config.checkpoint_dir is None:
        saver = NullSaver()
    else:
        checkpoint_path = Path(config.checkpoint_dir) / f"weights-{config.trial_id}.pth"
        saver = BestModelSaver(
            model, metric_name="mAP", num_checkpoints=1,
            checkpoint_path=str(checkpoint_path),
            mode="max", ema=True, alpha=0.2, verbose=True)

    writer = SummaryWriter(
        log_dir=str(Path(config.log_dir) / config.trial_id))

    def log_scalar(key, value, step):
        writer.add_scalar(key, value, step)

    def flush_log():
        writer.flush()

    dp_model.to(device)
    meters = defaultdict(partial(EMAMeter, alpha=0.2))
    for epoch in range(config.num_epochs):

        print(f"Epoch={epoch}")

        metrics = train_epoch(dp_model, train_loader, device, criterions, optimizer)
        log_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        for k, v in metrics.items():
            log_scalar(k, v, epoch)

        if (epoch == (config.num_epochs - 1)) or ((epoch + 1) % config.valid.step == 0):
            with torch.no_grad():
                metrics = validate_epoch(dp_model, val_loader, device, criterions, config.valid.score_threshold)
            for k, v in metrics.items():
                log_scalar("val_" + k, v, epoch)
                meters[k].update(v)

            print("[Metrics]")
            for k, v in metrics.items():
                print(f"  {k:10s}: {v:.4f}")
            saver.step(epoch, metrics)

        flush_log()

        lr_scheduler.step()

        print()

    metrics = {k: meter.get_value() for k, meter in meters.items()}
    return metrics


def train_epoch(model, data_loader, device, criterions, optimizer):
    print("  Training")
    model.train()
    meters = {k + "_loss": MeanMeter() for k in criterions.keys()}
    meters["loss"] = MeanMeter()
    progress = tqdm(data_loader)
    for i, sample in enumerate(progress):
        image = sample["image"].to(device)
        hm_true = sample["hm"].to(device)
        kp_mask = hm_true.max(axis=1, keepdim=True)[0].eq(1).float()
        num_keypoints = kp_mask.sum().clamp(1)
        pred = model(image)
        sum_loss = 0.0
        for key, (criterion, loss_weight) in criterions.items():
            pred_, true_ = pred[key], sample[key].to(device)
            loss = criterion(pred_, true_)
            if key != "hm":
                loss = (loss * kp_mask).sum() / num_keypoints / true_.size(1)  # reduce
            meters[key + "_loss"].update(loss.cpu().item())
            sum_loss += loss * loss_weight
        meters["loss"].update(sum_loss.cpu().item())
        optimizer.zero_grad()
        sum_loss.backward()
        optimizer.step()
        if i == (len(data_loader) - 1):
            progress.set_description(f"    loss={meters['loss'].get_value():.6f}")
        else:
            progress.set_description(f"    loss={sum_loss.item():.6f}")
    return {k: meter.get_value() for k, meter in meters.items()}


def validate_epoch(model, data_loader, device, criterions, score_threshold):
    print("  Validation")
    model.eval()
    meters = {k + "_loss": MeanMeter() for k in criterions.keys()}
    meters["loss"] = MeanMeter()
    meters["mAP"] = MeanMeter()
    num_samples = len(data_loader.dataset)
    scores = np.zeros(num_samples)
    image_ids = [None] * num_samples
    progress = tqdm(data_loader)
    sample_idx = 0
    for sample in progress:
        image = sample["image"].to(device)
        hm_true = sample["hm"].to(device)
        kp_mask = hm_true.max(axis=1, keepdim=True)[0].eq(1).float()
        num_keypoints = kp_mask.sum().clamp(1)
        pred = model(image)
        sum_loss = 0.0
        for key, (criterion, loss_weight) in criterions.items():
            pred_, true_ = pred[key], sample[key].to(device)
            loss = criterion(pred_, true_)
            if key != "hm":
                loss = (loss * kp_mask).sum() / num_keypoints / true_.size(1)  # reduce
            meters[key + "_loss"].update(loss.cpu().item())
            sum_loss += loss * loss_weight
        meters["loss"].update(sum_loss.cpu().item())

        for i, (image_id, bboxes_true) in enumerate(zip(sample["image_id"], sample["bboxes"])):
            bboxes_pred, confidences = get_bboxes(
                pred["hm"][i], pred["size"][i], pred["off"][i],
                score_threshold, limit=100)
            bboxes_true = coco_to_pascal(bboxes_true.to(bboxes_pred.device).to(bboxes_pred.dtype))
            bboxes_pred = coco_to_pascal(bboxes_pred)
            ap = sweep_average_precision(bboxes_true, bboxes_pred, confidences)
            meters["mAP"].update(ap)
            image_ids[sample_idx] = image_id
            scores[sample_idx] = ap
            sample_idx += 1

        if i == (len(data_loader) - 1):
            progress.set_description(f"    loss={meters['loss'].get_value():.6f}")
        else:
            progress.set_description(f"    loss={sum_loss.item():.6f}")

    # show worst
    print("[Worst Samples]")
    for i, sample_idx in enumerate(np.argsort(scores)[:5]):
        print(f"  #{i + 1}: {image_ids[sample_idx]} (AP={scores[sample_idx]:.4f})")

    return {k: meter.get_value() for k, meter in meters.items()}


def main():
    lr = 5e-4
    num_epochs = 128
    config = TrainConfig(
        trial_id="",
        device="cuda:0", seed=0,
        data=DataConfig(
            train_datasets=[
                DatasetConfig(
                    "kuzushiji",
                    "data/kuzushiji-recognition/train_images",
                    "data/kuzushiji-recognition/train.csv")
            ],
            val_datasets=[
                DatasetConfig(
                    "coco",
                    "data/test/preview",
                    "data/test/preview/20200611_coco_imglab.json")
            ],
            input_size=800,
            hm_alpha=1e-2,
            batch_size=2,
            seed=0
        ),
        model=ModelConfig(
            phi=0, pretrained=True,
            weights=None
        ),
        optimizer=OptimizerConfig(
            name="radam", config=dict(lr=5e-4)),
        scheduler=SchedulerConfig(
            name="cosine", config=dict(T_max=num_epochs - 1, eta_min=lr * 1e-1)),
        # scheduler=SchedulerConfig(
        #     name="step", config=dict(milestones=[])),
        loss=dict(hm=({}, 1.0), size=({}, 0.1), off=({}, 0.1)),
        num_epochs=num_epochs,
        valid=ValidConfig(
            step=1,
            score_threshold=0.01,
        ),
        checkpoint_dir="checkpoint",
        log_dir="tensorboard"
    )
    for phi in range(0, 7):
        config.trial_id = f"phi{phi}-{dt.now():%Y%m%d%H%M%S}"
        config.model.phi = phi
        train(config)


if __name__ == "__main__":
    main()
