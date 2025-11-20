import logging
import random

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        index,
        use_video_data=False,
        dataset_type="bss",
        target_sr=16000,
        limit=None,
        max_audio_length=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            text_encoder (CTCTextEncoder): text encoder.
            target_sr (int): supported sample rate.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            max_audio_length (int): maximum allowed audio length.
            max_test_length (int): maximum allowed text length.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self.use_video_data = use_video_data
        self.dataset_type = dataset_type

        self._assert_index_is_valid(index)

        index = self._filter_records_from_dataset(index, max_audio_length)
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        if not shuffle_index:
            index = self._sort_index(index)

        self._index: list[dict] = index

        self.target_sr = target_sr
        self.instance_transforms = instance_transforms

    def _bss_getitem(self, data_dict):
        instance_data = {"len": data_dict["len"]}

        for part in ["s1", "s2", "mix"]:
            path_key = f"{part}_path"
            audio_path = data_dict[path_key]
            audio = self.load_audio(audio_path)

            instance_data.update(
                {  # don't want to just copy data_dict because this may lead to bugs
                    part: audio,
                    path_key: audio_path,
                }
            )

        if self.use_video_data:
            for spk in ("s1", "s2"):
                mouth_key = f"{spk}_mouth_path"
                emb_key = f"{spk}_video_emb_path"

                if emb_key in data_dict:
                    emb_path = data_dict[emb_key]
                    emb = torch.load(emb_path, map_location="cpu")

                    instance_data.update(
                        {
                            f"{spk}_video_emb": emb,
                            emb_key: emb_path,
                        }
                    )
                elif mouth_key in data_dict:
                    mouth_path = data_dict[mouth_key]
                    npz = np.load(mouth_path)
                    frames = npz["data"]

                    video = torch.from_numpy(frames).unsqueeze(1)

                    instance_data.update(
                        {
                            f"{spk}_video": video,
                            mouth_key: mouth_path,
                        }
                    )
                else:
                    raise KeyError(
                        f"Missing both '{mouth_key}' and '{emb_key}' in index"
                    )
        return instance_data

    def _tss_getitem(self, data_dict, target_spk):
        instance_data = {"len": data_dict["len"]}

        mix_path = data_dict["mix_path"]
        target_path = data_dict[f"{target_spk}_path"]

        mix = self.load_audio(mix_path)
        target = self.load_audio(target_path)

        instance_data.update(
            {
                "mix": mix,
                "target": target,
                "mix_path": mix_path,
                "target_path": target_path,
            }
        )

        if self.use_video_data:
            mouth_key = f"{target_spk}_mouth_path"
            emb_key = f"{target_spk}_video_emb_path"

            if emb_key in data_dict:
                emb_path = data_dict[emb_key]
                emb = torch.load(emb_path, map_location="cpu")

                instance_data.update(
                    {
                        "video_emb": emb,
                        "video_emb_path": emb_path,
                    }
                )
            elif mouth_key in data_dict:
                mouth_path = data_dict[mouth_key]
                npz = np.load(mouth_path)
                frames = npz["data"]

                video = torch.from_numpy(frames).unsqueeze(1)

                instance_data.update(
                    {
                        "video": video,
                        "mouth_path": mouth_path,
                    }
                )
            else:
                raise KeyError(f"Missing both '{mouth_key}' and '{emb_key}' in index")

        return instance_data

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """

        if self.dataset_type == "tss":
            data_dict = self._index[ind // 2]
            instance_data = self._tss_getitem(data_dict, "s1" if ind % 2 == 0 else "s2")
        elif self.dataset_type == "bss":
            data_dict = self._index[ind]
            instance_data = self._bss_getitem(data_dict)
        else:
            raise ValueError("dataset_type can be one of ('tss', 'bss')")

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        if self.dataset_type == "bss":
            return len(self._index)
        elif self.dataset_type == "tss":
            return 2 * len(self._index)
        else:
            raise ValueError("dataset_type can be one of ('tss', 'bss')")

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data = self.instance_transforms[transform_name](instance_data)
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
        max_audio_length,
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        the desired max_audio_length.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            max_audio_length (int): maximum allowed audio length.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = (
                np.array([el["len"] for el in index]) >= max_audio_length
            )
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "  # noqa: E231
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        records_to_filter = (
            exceeds_audio_length  # TODO implement some video filtration if needed
        )

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total} ({_total / initial_size:.1%}) records from dataset"  # noqa: E231
            )

        return index

    def _assert_index_is_valid(self, index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            for part in ["s1", "s2", "mix"]:
                assert (
                    f"{part}_path" in entry
                ), "Each dataset item should include field '{part}_path' - path to {part} audio file."

            assert "len" in entry, (
                "Each dataset item should include field 'len'"
                " - length of the all audios."
            )

            if self.use_video_data:
                for part in ["s1", "s2"]:
                    assert (
                        f"{part}_mouth_path" in entry
                        or f"{part}_video_emb_path" in entry
                    ), (
                        f"When use video each dataset item should include one of fields: "
                        f"'{part}_mouth_path' - path to {part} mouth file or "
                        f"'{part}_video_emb_path' - path to {part} video embedding file"
                    )

    @staticmethod
    def _sort_index(index):
        """
        Sort index by audio length.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): sorted list, containing dict for each element
                of the dataset. The dict has required metadata information,
                such as label and object path.
        """
        return sorted(index, key=lambda x: x["len"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
