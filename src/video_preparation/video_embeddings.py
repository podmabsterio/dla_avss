from pathlib import Path

import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.video_collate import video_collate_fn
from src.datasets.video_dataset import VideoDataset
from src.model.RTFSNet.encoders import VideoEncoder


def _create_emb_dir(path: Path):
    print(f"Creating video embeddings directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _save_embs(embs, save_dir: Path, names):
    for emb, name in zip(embs, names):
        torch.save(emb, save_dir / f"{name}.pt")


@torch.no_grad()
def _generate_and_save_video_embeddings(
    dataloader, device, weights_dir, save_dir, video_enc_name
):
    video_encoder = VideoEncoder(weights_dir, video_enc_name, device)

    save_dir = Path(save_dir)
    is_dir_created = False
    for batch in tqdm(
        dataloader,
        desc="Generating video embeddings",
        total=len(dataloader),
    ):
        video_batch = batch["videos"].to(device)
        embs = video_encoder(video_batch)

        if not is_dir_created:
            is_dir_created = True
            _create_emb_dir(save_dir)
        _save_embs(embs, save_dir=save_dir, names=batch["video_names"])


def create_video_embeddings_if_needed(device, config):
    if not config.video_encoder.prepare_embeddings:
        print("Skipping embeddings creation")
        return

    emb_dir = Path(config.video_encoder.dataset_path) / "video_embeddings"
    dataset_dir = Path(config.video_encoder.dataset_path) / "mouths"
    if emb_dir.exists():
        print("Video embeddings directory exists, skipping embeddings creation")
        return

    dataset = VideoDataset(dataset_dir)

    dataloader = instantiate(
        config.video_encoder.dataloader,
        dataset=dataset,
        collate_fn=video_collate_fn,
        drop_last=False,
    )
    _generate_and_save_video_embeddings(
        dataloader,
        device,
        config.video_encoder.weights_dir,
        emb_dir,
        config.video_encoder.model_name,
    )
    print(f"Video embeddings created and saved to: {emb_dir}")
