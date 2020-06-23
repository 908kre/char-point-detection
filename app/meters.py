import numpy as np
import operator
import math
from logging import getLogger
from typing_extensions import Literal


class EMAMeter:
    def __init__(
        self, name: str = "", alpha: float = 0.99, log_period: int = 10
    ) -> None:
        self.alpha = alpha
        self.ema = np.nan
        self.logger = getLogger(name)
        self.count = 0
        self.log_period = log_period

    def update(self, value: float) -> None:
        if np.isnan(self.ema):
            self.ema = value
        else:
            self.ema = self.ema * self.alpha + value * (1.0 - self.alpha)
        self.count += 1
        if self.count % self.log_period == 0:
            self.count = 0
            self.logger.info(f"{self.ema:.4f}")

    def get_value(self) -> float:
        return self.ema

    def reset(self) -> None:
        self.ema = np.nan
        self.logger.info("reset")


class MeanMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0.0

    def update(self, value: float, count: int = 1) -> None:
        self.sum += value * count
        self.count += count

    def get_value(self) -> float:
        return self.sum / self.count

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0.0


class BestWatcher:
    def __init__(
        self,
        mode: Literal["min", "max"] = "min",
        min_delta: float = 0.0,
        ema: bool = False,
        alpha: float = 1.0,
    ) -> None:
        self.mode = mode
        self.min_delta = min_delta
        self.ema = ema
        self.alpha = alpha
        if mode == "min":
            self.op = operator.lt
            self.best = math.inf
        else:
            self.op = operator.gt
            self.best = -math.inf
        self.prev_metrics = math.nan

    def step(self, metrics: float) -> bool:
        if self.ema and not math.isnan(self.prev_metrics):
            metrics = self.prev_metrics * self.alpha + metrics * (1 - self.alpha)
        self.prev_metrics = metrics
        if self.op(metrics - self.min_delta, self.best):
            self.best = metrics
            return True
        else:
            return False
