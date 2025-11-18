from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from src.transforms.video import VideoNormalization


class VideoDataset(Dataset):
    def __init__(self, dataset_path: str = "data/avss_dataset/mouths", *args, **kwargs):
        """
        Args:
            dataset_path (str): path to dataset
            partition (str): partition name
            use_video_data (bool): if True tries to find video data in dataset directory
        """
        super().__init__()
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(
                "Can't find dataset with path: " + str(dataset_path.resolve())
            )

        self.video_files = self._find_files(dataset_path)
        self.normalize = VideoNormalization()

    def _find_files(self, dataset_path: Path):
        return [
            p
            for p in dataset_path.iterdir()
            if p.is_file() and not p.name.startswith(".")
        ]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        mouth_path: Path = self.video_files[index]
        npz = np.load(mouth_path)
        frames = npz["data"]

        video = torch.from_numpy(frames).unsqueeze(0)

        item_dict = {"video": video, "video_name": mouth_path.stem}

        return self.normalize(item_dict)
