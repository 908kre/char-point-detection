import torch
import typing as t
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch import nn
from pathlib import Path
import json
import random
import numpy as np
from torch import Tensor
from app.entities.box import CoCoBoxes, YoloBoxes, yolo_to_coco


def init_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed(seed)  # type: ignore


class DetectionPlot:
    def __init__(self, figsize: t.Tuple[int, int] = (4, 4)) -> None:
        self.w, self.h = (128, 128)
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.imshow(torch.ones(self.w, self.h, 3), interpolation="nearest")

    def __del__(self) -> None:
        plt.close(self.fig)

    def save(self, path: t.Union[str, Path]) -> None:
        self.fig.savefig(path)

    def with_image(self, image: Tensor) -> None:
        if len(image.shape) == 2:
            self.ax.imshow(image, interpolation="nearest")
            self.h, self.w = image.shape
        elif len(image.shape) == 3:
            _, self.h, self.w, = image.shape
            image = image.permute(1, 2, 0)
            self.ax.imshow(image, interpolation="nearest")
        else:
            shape = image.shape
            raise ValueError(f"invald shape={shape}")

    def with_yolo_boxes(
        self,
        boxes: YoloBoxes,
        probs: t.Optional[Tensor] = None,
        color: str = "black",
        fontsize: int = 7,
    ) -> None:
        self.with_coco_boxes(
            boxes=yolo_to_coco(boxes, size=(self.w, self.h)),
            probs=probs,
            color=color,
            fontsize=fontsize,
        )

    def with_coco_boxes(
        self,
        boxes: CoCoBoxes,
        probs: t.Optional[Tensor] = None,
        color: str = "black",
        fontsize: int = 7,
    ) -> None:
        """
        boxes: coco format
        """
        b, _ = boxes.shape
        _probs = probs if probs is not None else torch.ones((b,))
        _boxes = boxes.clone()
        for box, p in zip(_boxes, _probs):
            x0 = box[0]
            y0 = box[1]
            self.ax.text(x0, y0, f"{p:.2f}", fontsize=fontsize, color=color)
            rect = mpatches.Rectangle(
                (x0, y0),
                width=box[2],
                height=box[3],
                fill=False,
                edgecolor=color,
                linewidth=1,
            )
            self.ax.add_patch(rect)


class ModelLoader:
    def __init__(self, out_dir: str, model: nn.Module,) -> None:
        self.out_dir = Path(out_dir)
        self.checkpoint_file = self.out_dir / "checkpoint.json"
        self.model = model

        self.out_dir.mkdir(exist_ok=True)
        if self.checkpoint_file.exists():
            self.load()

    def load(self) -> None:
        with open(self.checkpoint_file, "r") as f:
            data = json.load(f)
        self.model.load_state_dict(
            torch.load(output_dir.joinpath(f"model.pth"))  # type: ignore
        )

    def save(self) -> None:
        torch.save(self.model.state_dict(), self.out_dir / f"model.pth")  # type: ignore
