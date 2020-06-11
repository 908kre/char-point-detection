import torch
import torch.nn as nn


def focal_loss(pred, gt, alpha=2.0, beta=4.0):
    pos_mask = gt.eq(1).float()
    neg_mask = gt.lt(1).float()
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
    neg_loss = -((1 - gt) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg_mask
    loss = (pos_loss + neg_loss).sum()
    num_pos = pos_mask.sum().clamp(1)
    return loss / num_pos.float()


class FocalLoss(nn.Module):

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        return focal_loss(input, target, self.alpha, self.beta)


class FocalLossWithLogits(nn.Module):

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        input = clamp_sigmoid(input)
        return focal_loss(input, target, self.alpha, self.beta)


def clamp_sigmoid(x, eps=1e-4):
    return torch.clamp(torch.sigmoid(x), min=eps, max=1 - eps)
