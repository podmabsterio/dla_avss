import torch
from torch import nn


class RMSNormalization(nn.Module):
    def __init__(self, eps=1e-8):
        super.__init__()
        self.eps = eps

    def forward(self, item):
        if "mix" not in item:
            raise ValueError("mix not in item")
        mix = item["mix"]
        energy = (mix**2).mean()
        scale = torch.sqrt(torch.clamp(energy, min=self.eps))
        gain = 1.0 / scale
        for key in ("s1", "s2", "mix"):
            if key in item:
                item[key] = item[key] * gain
        # TODO: video normalization
        item[
            "normalization_gain"
        ] = gain  # if we want to invert normalization for inference
        return item
