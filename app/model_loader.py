import torch
from logging import getLogger
from torch import nn
from pathlib import Path
from typing import Dict
import json

logger = getLogger(__name__)


class ModelLoader:
    def __init__(self, out_dir: str, model: nn.Module,) -> None:
        self.out_dir = Path(out_dir)
        self.checkpoint_file = self.out_dir / "checkpoint.json"
        self.model = model

        self.out_dir.mkdir(exist_ok=True, parents=True)
        if self.checkpoint_file.exists():
            self.load()

    def load(self) -> None:
        logger.info(f"load model from {self.out_dir}")
        with open(self.checkpoint_file, "r") as f:
            data = json.load(f)
        self.model.load_state_dict(
            torch.load(self.out_dir.joinpath(f"model.pth"))  # type: ignore
        )

    def save(self, metrics: Dict[str, float] = {}) -> None:
        with open(self.checkpoint_file, "w") as f:
            json.dump(metrics, f)
        logger.info(f"save model to {self.out_dir}")
        torch.save(self.model.state_dict(), self.out_dir / f"model.pth")  # type: ignore
