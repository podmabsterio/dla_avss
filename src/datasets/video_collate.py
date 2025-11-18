import torch


def video_collate_fn(dataset_items: list[dict]):
    videos = [item["video"] for item in dataset_items]
    video_names = [item["video_name"] for item in dataset_items]

    batched = {"videos": torch.stack(videos, dim=0), "video_names": video_names}

    return batched
