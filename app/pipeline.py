import torch
from ensemble_boxes import weighted_boxes_fusion
import numpy as np
from PIL import Image as PILImage
import cv2
from object_detection.utils import DetectionPlot
from cytoolz.curried import groupby, valmap, pipe, unique, map, reduce
from pathlib import Path
from typing import Any, List, Tuple, cast
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import ToTensor, Pad, ToPILImage
from torchvision.transforms.functional import crop
from sklearn.model_selection import StratifiedKFold
from object_detection.models.backbones.effnet import EfficientNetBackbone
from object_detection.metrics.mean_precision import MeanPrecition
from object_detection.models.centernetv1 import (
    CenterNetV1,
    Trainer as _Trainer,
    Visualize,
    Reg,
    ToBoxes,
    Criterion,
    MkGaussianMaps,
    MkFillMaps,
)
from torch.cuda.amp import GradScaler, autocast
import albumentations as albm
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
from object_detection.entities import (
    TrainSample,
    ImageBatch,
    YoloBoxes,
    ImageId,
    Labels,
    yolo_to_pascal,
    PascalBoxes,
    Confidences,
    pascal_to_yolo,
    Image,
)
from object_detection.model_loader import ModelLoader, BestWatcher
from app import config
from torch.utils.data import DataLoader, Subset, ConcatDataset
from .data.common import imread
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
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lr_scheduler = CosineAnnealingLR(
            optimizer=self.optimizer, T_max=config.T_max, eta_min=config.eta_min
        )

    def __call__(self, epochs: int) -> None:
        self.model = self.model_loader.load_if_needed(self.model)
        for epoch in range(epochs):
            self.train_one_epoch()

    def train_one_epoch(self) -> None:
        loader = self.train_loader
        for i, (images, box_batch, ids, _) in enumerate(tqdm(loader)):
            self.model.train()
            self.optimizer.zero_grad()
            images, box_batch = self.preprocess((images, box_batch))
            outputs = self.model(images)
            loss, hm_loss, box_loss, gt_hms = self.criterion(images, outputs, box_batch)
            loss.backward()
            self.optimizer.step()
            self.meters["train_loss"].update(loss.item())
            self.meters["train_box"].update(box_loss.item())
            self.meters["train_hm"].update(hm_loss.item())
            if i % 50 == 0:
                self.eval_one_epoch()
                self.lr_scheduler.step()
                self.model_loader.save_if_needed(
                    self.model, self.meters[self.model_loader.key].get_value()
                )
                self.log()
                self.reset_meters()


def train() -> None:
    coco = CocoDataset(
        image_dir="/store/datasets/preview",
        annot_file="/store/datasets/preview/20200611_coco_imglab.json",
        max_size=config.max_size,
    )
    codh = CodhKuzushijiDataset(
        image_dir="/store/codh-kuzushiji/resized",
        annot_file="/store/codh-kuzushiji/resized/annot.json",
        max_size=config.max_size,
    )
    neg = NegativeDataset(image_dir="/store/negative/images", max_size=config.max_size,)
    train_loader: Any = DataLoader(
        ConcatDataset([coco] * 100),  # type: ignore
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    test_loader: Any = DataLoader(
        ConcatDataset([coco, coco]),
        batch_size=config.batch_size * 2,
        drop_last=False,
        shuffle=False,
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
    visualize = Visualize(
        config.out_dir, "centernet", limit=10, use_alpha=True, figsize=(20, 20)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,)
    get_score = MeanPrecition(iou_thresholds=config.iou_thresholds)
    to_boxes = ToBoxes(threshold=config.confidence_threshold, use_peak=config.use_peak,)
    criterion = Criterion(
        heatmap_weight=config.heatmap_weight,
        box_weight=config.box_weight,
        mkmaps=MkGaussianMaps(sigma=config.sigma, mode=config.mode,),
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        model_loader=model_loader,
        optimizer=optimizer,
        visualize=visualize,
        device=config.device,
        get_score=get_score,
        criterion=criterion,
        to_boxes=to_boxes,
    )
    trainer(1000)


def img_to_batch(
    pil_image: PILImage, patch_size: int, overlap_size: int = 0,
) -> Tuple[ImageBatch, Tuple[int, int]]:
    w, h = pil_image.size
    pad_size = (0, 0, patch_size - w % patch_size, patch_size - h % patch_size)
    pil_image = Pad(pad_size)(pil_image)
    w, h = pil_image.size
    row_count = h // patch_size
    col_count = w // patch_size
    tensors = []
    to_tensor = ToTensor()
    for i in range(row_count):
        for j in range(col_count):
            tensors.append(
                to_tensor(
                    crop(
                        pil_image,
                        i * patch_size,
                        j * patch_size,
                        patch_size + overlap_size,
                        patch_size + overlap_size,
                    )
                )
            )
    return ImageBatch(torch.stack(tensors)), (row_count, col_count)


def shift_pascal(boxes: PascalBoxes, offset: Tuple[int, int]) -> PascalBoxes:
    if len(boxes) == 0:
        return boxes

    x0, y0, x1, y1 = boxes.unbind(-1)
    offset_x, offset_y = offset
    x0 = x0 + offset_x
    x1 = x1 + offset_x
    y0 = y0 + offset_y
    y1 = y1 + offset_y
    return PascalBoxes(torch.stack([x0, y0, x1, y1], dim=1))


def batch_to_image(
    batch: ImageBatch,
    box_batch: List[YoloBoxes],
    score_batch: List[Confidences],
    patch_size: int,
    batch_layout: Tuple[int, int],
    overlap_size: int = 0,
) -> Tuple[Image, YoloBoxes]:
    row_count, col_count = batch_layout
    device = batch.device
    out_image = torch.zeros(
        (
            3,
            row_count * patch_size + overlap_size,
            col_count * patch_size + overlap_size,
        )
    ).to(device)
    _, h, w = out_image.shape
    out_box_batch = []
    out_score_batch = []
    labels_list = []
    for i in range(row_count):
        for j in range(col_count):
            idx = col_count * i + j
            patch_img = batch[idx]
            offset_y = i * patch_size
            offset_x = j * patch_size
            boxes = shift_pascal(
                yolo_to_pascal( box_batch[idx], (patch_size + overlap_size, patch_size + overlap_size),),
                (offset_x, offset_y),
            )
            boxes[:,[0, 2]] = boxes[:, [0, 2]] / w
            boxes[:,[1, 3]] = boxes[:, [1, 3]] / h
            out_box_batch.append(boxes)
            out_score_batch.append(score_batch[idx])
            labels_list.append(torch.zeros(len(boxes)))
            out_image[
                :,
                offset_y : offset_y + patch_size + overlap_size,
                offset_x : offset_x + patch_size + overlap_size,
            ] = patch_img
    m_boxes, m_scores, _ = weighted_boxes_fusion(out_box_batch, out_score_batch, labels_list=labels_list, iou_thr=0.9)
    out_boxes = pascal_to_yolo(PascalBoxes(torch.tensor(m_boxes)), (1, 1))
    out_scores = Confidences(torch.tensor(m_scores))
    return out_image.cpu(), out_boxes, out_scores


@torch.no_grad()
def submit() -> None:
    device = torch.device("cuda")

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
    model = cast(CenterNetV1, model_loader.load_if_needed(model)).to(device)
    to_boxes = ToBoxes(threshold=config.confidence_threshold, use_peak=config.use_peak,)
    # overlap_size = config.max_size // 2
    # patch_size = config.max_size - overlap_size
    pil_image = PILImage.open("/store/datasets/preview/RFnSdvHDYb3TkYpJ.jpg")

    transform = albm.Compose([
        albm.LongestMaxSize(max_size=config.max_size),
        albm.PadIfNeeded(
            min_width=config.max_size,
            min_height=config.max_size,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        ToTensorV2(),
    ])
    img = transform(image=np.array(pil_image))['image']

    image_batch = (img.float()/255).unsqueeze(0).to(device)
    netout = model(image_batch)
    preds = to_boxes(netout)
    plot = DetectionPlot(h=config.max_size, w=config.max_size, use_alpha=True, figsize=(20, 20), show_probs=True,)
    plot.with_image(image_batch[0])
    plot.with_yolo_boxes(preds[0][0], probs=preds[1][0], color="green")
    plot.save(f"/store/pred-boxes.png")
