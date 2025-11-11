import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm
from pathlib import Path

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class AVSSDataset(BaseDataset):
    def __init__(
        self, partition: str, dataset_path: str='data/avss_dataset', use_video_data: bool=False, force_reindex=False, *args, **kwargs
    ):
        """
        Args:
            dataset_path (str): path to dataset
            partition (str): partition name
            use_video_data (bool): if True tries to find video data in dataset directory
        """
        index_path = ROOT_PATH / "data" / "avss_dataset" / "audio" / partition / "index.json"
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError("Can't find dataset with path: " + str(dataset_path.resolve()))
            
        if index_path.exists() and not force_reindex:
            index = read_json(str(index_path))
        else:
            index = self._create_index(dataset_path, partition, use_video_data, index_path)
        super().__init__(index, *args, use_video_data=use_video_data, **kwargs)

    def _create_index(self, dataset_path: Path, partition: str, use_video_data: bool, index_path: Path):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            dataset_path (Path): path to dataset
            partition (str): partition name
            use_video_data (bool): if True tries to find video data in dataset directory
            index_path (Path): path to save index
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        
        data_path = dataset_path / 'audio' / partition
        s1_path = data_path / 's1'
        s2_path = data_path / 's2'
        mix_path = data_path / 'mix'
        
        mouths_path = dataset_path / 'mouths'

        tqdm.write("Creating index...")
        mix_files = [p for p in Path(mix_path).iterdir() if p.is_file() and not p.name.startswith('.')]
        
        for mix_file in tqdm(mix_files, total=len(mix_files), desc='Indexing files'):
            s1_file = s1_path / mix_file.name
            s2_file = s2_path / mix_file.name
            
            if not s1_file.exists():  # TODO this dataset is not suitable for inference
                raise FileNotFoundError("Can't find s1 file for following mix file\nmix file path:         " + \
                                        str(mix_file.resolve()) + \
                                        "\nexpected s1 file path: " + \
                                        str(s1_file.resolve()))
            
            if not s2_file.exists():
                raise FileNotFoundError("Can't find s2 file for following mix file\nmix file path:         " + \
                                        str(mix_file.resolve()) + \
                                        "\nexpected s2 file path: " + \
                                        str(s2_file.resolve()))
            
            s1_info = torchaudio.info(str(s1_file))
            s2_info = torchaudio.info(str(s2_file))
            mix_info = torchaudio.info(str(mix_file))
            
            s1_length = s1_info.num_frames / s1_info.sample_rate
            s2_length = s2_info.num_frames / s2_info.sample_rate
            mix_length = mix_info.num_frames / mix_info.sample_rate
            
            if not (s1_length == s2_length and s2_length == mix_length):
                raise ValueError(f'Length of audios with id {s1_file.stem} are inconsistent')
            
            index_element = {
                "s1_path": str(s1_file), 
                "s2_path": str(s2_file), 
                "mix_path": str(mix_file),
                "len": s1_length,
            }
            
            if use_video_data:
                # TODO make index_element.update(...) with video fields
                raise NotImplementedError('Videos are not supported')
            
            index.append(index_element)
            
            

        write_json(index, index_path)

        return index
