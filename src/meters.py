import operator
import math
import numpy as np


class EMAMeter:
    def __init__(self, alpha=0.99):
        self.alpha = alpha
        self.ema = np.nan

    def update(self, value):
        if np.isnan(self.ema):
            self.ema = value
        else:
            self.ema = self.ema * self.alpha + value * (1.0 - self.alpha)

    def get_value(self):
        return self.ema

    def reset(self):
        self.ema = np.nan


class MeanMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0.0

    def update(self, value, count=1):
        self.sum += value * count
        self.count += count

    def get_value(self):
        return self.sum / self.count

    def reset(self):
        self.sum = 0.0
        self.count = 0.0


class BestWatcher:
    def __init__(self, mode="min", min_delta=0.0, ema=False, alpha=1.0):
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

    def step(self, metrics):
        if self.ema and not math.isnan(self.prev_metrics):
            metrics = self.prev_metrics * self.alpha + metrics * (1 - self.alpha)
        self.prev_metrics = metrics
        if self.op(metrics - self.min_delta, self.best):
            self.best = metrics
            return True
        else:
            return False
