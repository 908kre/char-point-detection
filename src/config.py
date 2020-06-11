from dataclasses import dataclass
from typing import Dict, Tuple, List


@dataclass
class TrainConfig:
    trial_id: str
    device: str
    seed: int
    data: "DataConfig"
    model: "ModelConfig"
    optimizer: "OptimizerConfig"
    scheduler: "SchedulerConfig"
    loss: Dict[str, Tuple[dict, float]]
    num_epochs: int
    valid: "ValidConfig"
    checkpoint_dir: str
    log_dir: str


@dataclass
class TestConfig:
    device: str
    data: "DataConfig"
    model: "ModelConfig"
    score_threshold: float


@dataclass
class DataConfig:
    train_datasets: List["DatasetConfig"]
    val_datasets: List["DatasetConfig"]
    input_size: int
    hm_alpha: float
    batch_size: int
    seed: int


@dataclass
class DatasetConfig:
    name: str
    image: str
    annot: str


@dataclass
class ModelConfig:
    phi: str
    pretrained: bool
    weights: str


@dataclass
class ValidConfig:
    step: int
    score_threshold: float


@dataclass
class OptimizerConfig:
    name: str
    config: dict


@dataclass
class SchedulerConfig:
    name: str
    config: dict
