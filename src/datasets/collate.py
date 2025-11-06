import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    parts = (
        ["s1", "s2", "mix"] if "s1" in dataset_items[0] else ["mix"]
    )  # for inference

    audios = {part: [] for part in parts}
    paths = {part: [] for part in parts}

    audio_lens = []

    audio_pad = 0

    for item in dataset_items:
        audio_lens.append(item["mix"].size(1))

        for part in parts:
            audios[part].append(item[part])
            paths[part].append(item[f"{part}_path"])

    L = int(max(audio_lens))
    B = len(dataset_items)
    dtype = audios["mix"][0].dtype

    result_batch = {
        "audio_lens": torch.tensor(audio_lens, dtype=torch.long),
    }

    mix_batch = torch.full((B, 1, L), fill_value=audio_pad, dtype=dtype)
    for i, a in enumerate(audios["mix"]):
        t = a.size(1)
        mix_batch[i, 0, :t].copy_(a.squeeze(0))

    if "s1" in parts:
        target_batch = torch.full((B, 2, L), fill_value=audio_pad, dtype=dtype)
        for i, (s1, s2) in enumerate(zip(audios["s1"], audios["s2"])):
            t = a.size(1)
            target_batch[i, 0, :t].copy_(s1.squeeze(0))
            target_batch[i, 1, :t].copy_(s2.squeeze(0))

    result_batch["mix"] = mix_batch
    result_batch["target"] = target_batch
    for part in parts:
        result_batch[f"{part}_paths"] = paths[part]

    return result_batch
