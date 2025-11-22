import os
from typing import List, Tuple

import torch
from torch import Tensor
from torch_audiomentations import AddBackgroundNoise


class RandomBackgroundNoise:
    def __init__(
        self,
        noises_dir: str,
        sample_rate: int = 16000,
        p_apply: float = 1.0,
        snr_db_range: Tuple[float, float] = (0.0, 30.0),
    ):
        self.sample_rate = sample_rate
        self.p_apply = p_apply
        self.min_snr_db = snr_db_range[0]
        self.max_snr_db = snr_db_range[1]

        noise_paths: List[str] = []
        for root, _, files in os.walk(noises_dir):
            for fname in files:
                if fname.lower().endswith(".wav"):
                    noise_paths.append(os.path.join(root, fname))

        if not noise_paths:
            raise RuntimeError(
                f"No .wav files found in noises_dir='{noises_dir}'. "
                f"Check path and dataset structure."
            )

        self.transform = AddBackgroundNoise(
            background_paths=noise_paths,
            min_snr_in_db=self.min_snr_db,
            max_snr_in_db=self.max_snr_db,
            mode="per_example",
            p=1.0,
            sample_rate=self.sample_rate,
            output_type="tensor",
        )

        self.transform.train()

    def __call__(self, item: Tensor) -> dict:
        if torch.rand(1).item() > self.p_apply:
            return item

        mix = item["mix"]
        mix = mix.unsqueeze(0)

        augmented = self.transform(mix, sample_rate=self.sample_rate)
        augmented = torch.clamp(augmented, -1.0, 1.0)
        item["mix"] = augmented.squeeze(0)

        return item
