import os
from typing import Dict

import torch
import torchaudio
from torch.utils.data import Dataset


class EvalBssDataset(Dataset):
    def __init__(self, gt_dir: str, pred_dir: str):
        super().__init__()
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir

        self.gt_s1_dir = os.path.join(gt_dir, "s1")
        self.gt_s2_dir = os.path.join(gt_dir, "s2")
        self.pr_s1_dir = os.path.join(pred_dir, "s1")
        self.pr_s2_dir = os.path.join(pred_dir, "s2")

        gt_s1_files = sorted(os.listdir(self.gt_s1_dir))
        pr_s1_files = sorted(os.listdir(self.pr_s1_dir))

        if gt_s1_files != pr_s1_files:
            raise RuntimeError("File names mismatch between GT and PRED in s1")

        gt_s2_files = sorted(os.listdir(self.gt_s2_dir))
        pr_s2_files = sorted(os.listdir(self.pr_s2_dir))

        if gt_s2_files != pr_s2_files:
            raise RuntimeError("File names mismatch between GT and PRED in s2")

        self.files_s1 = gt_s1_files
        self.files_s2 = gt_s2_files

    def __len__(self) -> int:
        return len(self.files_s1)

    def _load_audio(self, path: str) -> torch.Tensor:
        wav, _ = torchaudio.load(path)
        return wav

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gt_s1_path = os.path.join(self.gt_s1_dir, self.files_s1[idx])
        pr_s1_path = os.path.join(self.pr_s1_dir, self.files_s1[idx])

        gt_s2_path = os.path.join(self.gt_s2_dir, self.files_s2[idx])
        pr_s2_path = os.path.join(self.pr_s2_dir, self.files_s2[idx])

        gt_s1 = self._load_audio(gt_s1_path)
        pr_s1 = self._load_audio(pr_s1_path)

        gt_s2 = self._load_audio(gt_s2_path)
        pr_s2 = self._load_audio(pr_s2_path)

        target = torch.cat([gt_s1, gt_s2], dim=0)
        preds = torch.cat([pr_s1, pr_s2], dim=0)

        return {"target": target, "preds": preds}
