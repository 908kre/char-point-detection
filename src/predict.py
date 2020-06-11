from argparse import ArgumentParser
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.ops.boxes import nms
from tqdm import tqdm
import cv2
import albumentations as albm
import albumentations.pytorch as albm_torch
from src.config import ModelConfig
from src.data import NORMALIZE_PARAMS, ImageDataset, collate_fn
from src.transforms import MakeMap
from src.models import get_model
from src.models.ctdet import get_bboxes
from src.util import coco_to_pascal


def main():
    args = parse_args()
    device = torch.device(args.device)
    loader = get_loader(args.image_dir, args.input_size, args.batch_size)
    model = get_model(ModelConfig(phi=0, pretrained=True, weights=args.weights))
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    results = []
    for batch in tqdm(loader):
        x = batch["image"].to(device)
        num_samples = len(x)
        with torch.no_grad():
            y = model(x)
            for i in range(num_samples):
                image_id = batch["image_id"][i]
                image_size = batch["image_size"][i]
                boxes, scores = get_bboxes(y["hm"][i], y["size"][i], y["off"][i], args.score_threshold)
                keep = nms(coco_to_pascal(boxes), scores, args.nms_threshold)
                boxes = restore_boxes(image_size, args.input_size, boxes[keep].cpu().numpy())
                scores = scores[keep].cpu().numpy()
                results.append({
                    "image_id": image_id,
                    "boxes": boxes.tolist(),
                    "scores": scores.tolist()
                })
    with open(args.output_file, "w") as fp:
        json.dump(results, fp)


def restore_boxes(src_size, dst_size, boxes):
    src_longer = max(src_size)
    boxes *= src_longer / dst_size  # scale
    boxes[:, :2] -= (src_longer - np.array(src_size)) / 2  # unpad
    return boxes


def get_loader(image_dir: str, input_size: int, batch_size: int):
    transforms = albm.Compose([
        albm.LongestMaxSize(input_size),
        albm.PadIfNeeded(
            input_size, input_size,
            border_mode=cv2.BORDER_CONSTANT, value=0),
        albm.Normalize(**NORMALIZE_PARAMS),
        albm_torch.ToTensorV2()
    ])
    dataset = ImageDataset(
        image_dir,
        transforms=lambda x: transforms(**x)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size, drop_last=False, shuffle=False,
        collate_fn=collate_fn,
        num_workers=10, pin_memory=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--weights", default="weights.pth")
    parser.add_argument("--image-dir", default="data/test/preview")
    parser.add_argument("--output-file", default="pred.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.01)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
