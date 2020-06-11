from typing import List
import operator
import math
import json
import contextlib
from pathlib import Path
import torch


class NullSaver:

    def __init__(self):
        self.enabled = True

    def step(self, epoch: int, metrics: dict):
        pass


class BestModelSaver:

    def __init__(self, model, metric_name="loss", num_checkpoints=3, wait=5, checkpoint_path=None,
                 mode="min", min_delta=0., ema=False, alpha=1., verbose=True, enabled=True):
        self.model = model
        self.metric_name = metric_name
        self.num_checkpoints = num_checkpoints
        self.wait = wait
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.min_delta = min_delta
        self.ema = ema
        self.alpha = alpha
        self.verbose = verbose
        if mode == "min":
            self.op = operator.lt
        else:
            self.op = operator.gt
        self.prev_value = math.nan
        self.current = None  # type: CheckPoint
        self.history = []  # type: List[CheckPoint]
        self.enabled = enabled

    def step(self, epoch: int, metrics: dict):
        if not self.enabled:
            return False
        updated = False
        value = metrics[self.metric_name]
        if self.ema and not math.isnan(self.prev_value):
            value = self.prev_value * self.alpha + value * (1 - self.alpha)
        self.prev_value = value

        if (self.current is not None) and ((epoch - self.current.epoch) > self.wait):
            self.history.append(self.current)
            self.current = None

        if self.current is None:
            if len(self.history) < self.num_checkpoints:
                self._update_current(epoch, value, metrics)
                updated = True
            else:
                history = sorted(self.history, key=lambda _: _.value, reverse=self.mode != "min")
                worst = history[-1]
                if self.op(value - self.min_delta, worst.value):
                    if self.verbose:
                        print(f"[{self.metric_name}] delete checkpoint: epoch={worst.epoch}, value={worst.value:.3f}")
                    worst.delete()
                    self.history = history[:-1]
                    self._update_current(epoch, value, metrics)
                    updated = True
        elif self.op(value - self.min_delta, self.current.value):
            self._update_current(epoch, value, metrics)
            updated = True
        return updated

    def _update_current(self, epoch: int, value: float, metrics: dict=None):
        if self.current is not None:
            if self.verbose:
                print(f"[{self.metric_name}] delete checkpoint: epoch={self.current.epoch}, value={self.current.value:.3f}")
            self.current.delete()
        if self.verbose:
            print(f"[{self.metric_name}] new checkpoint: epoch={epoch}, value={value:.3f}")
        path = Path(self.checkpoint_path.format(epoch=epoch)) if self.checkpoint_path is not None else None
        stat = self.model.state_dict()
        checkpoint = CheckPoint(epoch, value, stat, path, metrics)
        checkpoint.save()
        self.current = checkpoint


class CheckPoint:

    def __init__(self, epoch: int, value: float, weights: dict, path: Path, metrics: dict=None):
        self.epoch = epoch
        self.value = value
        self.weights = weights
        self.path = path
        self.metrics = metrics
    
    def save(self, path=None):
        if path is None:
            path = self.path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.weights, path)
            _save_json(
                self.metrics.update(dict(epoch=self.epoch)),
                self._metrics_path(path))

    def delete(self):
        if self.path is not None:
            self.path.unlink()
        metrics_path = self._metrics_path(self.path)
        if metrics_path is not None:
            with contextlib.suppress(FileNotFoundError):
                metrics_path.unlink()

    def _metrics_path(self, path):
        return None if path is None else path.with_suffix(".json")


def _save_json(obj, path):
    if (obj is None) or (path is None):
        return
    with open(path, "w") as fp:
        json.dump(obj, fp)
