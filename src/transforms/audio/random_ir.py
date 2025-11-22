import os
from typing import List

import torch
from torch import Tensor
from torch_audiomentations import ApplyImpulseResponse


class RandomImpulseResponse:
    def __init__(
        self,
        irs_dir: str,
        sample_rate: int = 16000,
        p_apply: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.p_apply = p_apply

        ir_paths: List[str] = []
        for root, _, files in os.walk(irs_dir):
            for fname in files:
                if fname.lower().endswith(".wav"):
                    ir_paths.append(os.path.join(root, fname))

        if not ir_paths:
            raise RuntimeError(
                f"No .wav files found in irs_dir='{irs_dir}'. "
                f"Check path and dataset structure."
            )

        self.transform = ApplyImpulseResponse(
            ir_paths=ir_paths,
            mode="per_example",
            p=1.0,
            sample_rate=self.sample_rate,
            output_type="tensor",
        )

        self.transform.train()

    def __call__(self, item: dict) -> dict:
        if torch.rand(1).item() > self.p_apply:
            return item

        mix: Tensor = item["mix"]
        target: Tensor = item["target"]

        combined = torch.cat([mix, target], dim=0)
        combined = combined.unsqueeze(0)

        try:
            augmented = self.transform(combined, sample_rate=self.sample_rate)
        except Exception:
            return item

        augmented = torch.clamp(augmented, -1.0, 1.0)

        augmented = augmented.squeeze(0)

        augmented_mix = augmented[:1, :]
        augmented_target = augmented[1:, :]

        item["mix"] = augmented_mix
        item["target"] = augmented_target

        return item
