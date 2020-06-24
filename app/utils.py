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


def init_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed(seed)  # type: ignore
