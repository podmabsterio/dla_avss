import torch.nn as nn


class PeakNormalization(nn.Module):
    def __init__(self, peak_target: float = 0.99, eps: float = 1e-8):
        super().__init__()
        self.peak_target = peak_target
        self.eps = eps

    def forward(self, item: dict):
        if "mix" not in item:
            raise ValueError("mix not in item")

        x = item["mix"]
        peak = x.abs().amax().clamp_min(self.eps)
        gain = self.peak_target / peak

        for key in ("mix", "s1", "s2"):
            if key in item and item[key] is not None:
                item[key] = item[key] * gain

        # TODO: video normalization
        item[
            "normalization_gain"
        ] = gain  # if we want to invert normalization for inference
        return item
