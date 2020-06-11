import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterNetDetector(nn.Module):

    def __init__(self, backbone: nn.Module, fusion: nn.Module, hidden_channels=64):
        super().__init__()
        self.backbone = backbone
        self.fusion = fusion
        self.hm_head = PointwiseRegressor(fusion.out_channels, hidden_channels, 1)
        self.size_head = PointwiseRegressor(fusion.out_channels, hidden_channels, 2)
        self.off_head = PointwiseRegressor(fusion.out_channels, hidden_channels, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fusion(x[1:])
        hm = self.hm_head(x)
        size = self.size_head(x)
        off = self.off_head(x)
        return dict(hm=hm, size=size, off=off)


class PointwiseRegressor(nn.Module):

    def __init__(self, in_channels=1024, hidden_channels=64, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.fc = nn.Conv2d(hidden_channels, out_channels, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.fc(x)
        return x


def get_bboxes(hm, size, off, score_threshold, limit=5000):
    confidences = []
    bboxes = []
    hm = torch.sigmoid(hm)
    kp_map = (F.max_pool2d(hm, 3, stride=1, padding=1) == hm) & (hm > score_threshold)
    indices = kp_map[0].nonzero()
    confidences_ = hm[0][indices[:, 0], indices[:, 1]]
    sort_idx = confidences_.argsort(descending=True)[:limit]
    indices, confidences_ = indices[sort_idx], confidences_[sort_idx]
    offsets = off[:, indices[:, 0], indices[:, 1]].t()  # dcx, dcy
    loc = (indices.flip(1) + offsets.clamp(0, 1)) * 4  # cx, cy
    sizes = size[:, indices[:, 0], indices[:, 1]].t()  # w, h
    bb = torch.cat([
        loc - sizes / 2,  # cx, cy -> x, y
        sizes  # w, h
    ], dim=1)
    confidences.append(confidences_)
    bboxes.append(bb)
    return torch.cat(bboxes), torch.cat(confidences)
