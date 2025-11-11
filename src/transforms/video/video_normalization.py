import torch
from torch import nn


class VideoNormalization(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, item):
        for key in ("s1_video", "s2_video"):
            if key not in item:
                continue

            video = item[key]

            if not torch.is_floating_point(video):
                video = video.float() / 255.0

            mean = video.mean()
            std = video.std().clamp(min=self.eps)
            video = (video - mean) / std

            item[key] = video

        return item
