from app.models import NNModel
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Trainer:
    def __init__(self,) -> None:
        self.model = NNModel().to(device)
