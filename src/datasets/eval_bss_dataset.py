import os
from typing import Callable, Dict, Optional

import torch
import torchaudio
from torch.utils.data import Dataset

from src.datasets.data_utils import apply_instance_transorms


class EvalBssDataset(Dataset):
    def __init__(
        self,
        gt_dir: str,
        pred_dir: str,
        instance_transforms: Optional[Dict[str, Callable]] = None,
    ):
        super().__init__()
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir

        self.gt_s1_dir = os.path.join(gt_dir, "s1")
        self.gt_s2_dir = os.path.join(gt_dir, "s2")
        self.gt_mix_dir = os.path.join(gt_dir, "mix")

        self.pr_s1_dir = os.path.join(pred_dir, "s1")
        self.pr_s2_dir = os.path.join(pred_dir, "s2")

        gt_s1_files = sorted(os.listdir(self.gt_s1_dir))
        gt_s2_files = sorted(os.listdir(self.gt_s2_dir))
        gt_mix_files = sorted(os.listdir(self.gt_mix_dir))

        pr_s1_files = sorted(os.listdir(self.pr_s1_dir))
        pr_s2_files = sorted(os.listdir(self.pr_s2_dir))

        if not (gt_s1_files == gt_s2_files == gt_mix_files):
            raise RuntimeError("File names mismatch inside GT folders (mix, s1, s2)")

        if gt_s1_files != pr_s1_files:
            raise RuntimeError("File names mismatch between GT and PRED in s1")

        if gt_s2_files != pr_s2_files:
            raise RuntimeError("File names mismatch between GT and PRED in s2")

        self.files = gt_s1_files
        self.instance_transforms = instance_transforms

    def __len__(self) -> int:
        return len(self.files)

    def _load_audio(self, path: str) -> torch.Tensor:
        wav, _ = torchaudio.load(path)
        return wav

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fname = self.files[idx]

        gt_s1_path = os.path.join(self.gt_s1_dir, fname)
        gt_s2_path = os.path.join(self.gt_s2_dir, fname)
        gt_mix_path = os.path.join(self.gt_mix_dir, fname)

        pr_s1_path = os.path.join(self.pr_s1_dir, fname)
        pr_s2_path = os.path.join(self.pr_s2_dir, fname)

        gt_s1 = self._load_audio(gt_s1_path)
        gt_s2 = self._load_audio(gt_s2_path)
        mix = self._load_audio(gt_mix_path)

        pr_s1 = self._load_audio(pr_s1_path)
        pr_s2 = self._load_audio(pr_s2_path)

        target = torch.cat([gt_s1, gt_s2], dim=0)
        preds = torch.cat([pr_s1, pr_s2], dim=0)

        instance_data = {
            "mix": mix,
            "target": target,
            "preds": preds,
        }

        return apply_instance_transorms(self.instance_transforms, instance_data)
