import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from dataset.__getitem__.

    Returns:
        result_batch (dict[Tensor]): dict with batched tensors.
    """
    parts = ["s1", "s2", "mix"] if "s1" in dataset_items[0] else ["mix"]

    audios = {part: [] for part in parts}
    videos = {part: [] for part in parts}
    embs = {part: [] for part in parts}
    audio_paths = {part: [] for part in parts}
    video_paths = {part: [] for part in parts}
    emb_paths = {part: [] for part in parts}

    audio_lens = []
    audio_pad = 0

    for item in dataset_items:
        audio_lens.append(item["mix"].size(1))

        for part in parts:
            audios[part].append(item[part])
            audio_paths[part].append(item[f"{part}_path"])

            video_key = f"{part}_video"
            mouth_path_key = f"{part}_mouth_path"
            emb_key = f"{part}_video_emb"
            emb_path_key = f"{part}_video_emb_path"

            if video_key in item:
                v = item[video_key]
                assert v.shape == (
                    50,
                    1,
                    96,
                    96,
                ), f"Video shape {v.shape} != (50, 1, 96, 96)"  # very straight forward assert, but it fits the task
                videos[part].append(v)
                video_paths[part].append(item[mouth_path_key])
            if emb_key in item:
                emb = item[emb_key]
                embs[part].append(emb)
                emb_paths[part].append(item[emb_path_key])

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
    result_batch["mix"] = mix_batch

    if "s1" in parts:
        target_batch = torch.full((B, 2, L), fill_value=audio_pad, dtype=dtype)
        for i, (s1, s2) in enumerate(zip(audios["s1"], audios["s2"])):
            t = s1.size(1)
            target_batch[i, 0, :t].copy_(s1.squeeze(0))
            target_batch[i, 1, :t].copy_(s2.squeeze(0))
        result_batch["target"] = target_batch

    for part in parts:
        result_batch[f"{part}_paths"] = audio_paths[part]

    for part in parts:
        if len(videos[part]) > 0:
            result_batch[f"{part}_video"] = torch.stack(videos[part], dim=0)
            result_batch[f"{part}_mouth_paths"] = video_paths[part]
        if len(embs[part]) > 0:
            result_batch[f"{part}_video_emb"] = torch.stack(embs[part], dim=0)
            result_batch[f"{part}_video_emb_paths"] = emb_paths[part]

    return result_batch
