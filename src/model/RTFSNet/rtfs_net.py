import torch
from torch import nn

from src.model.RTFSNet.encoders import AudioEncoder, VideoEncoder


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(z: torch.Tensor):
        """
        Args:
            z (torch.Tensor): tensor a_R from paper with shape (B, C_a, T_a, F)
        Outputs:
            separated_audio (torch.Tensor): separated audio batch
        """
        raise NotImplementedError()


class RTFSNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, video_emb: torch.Tensor):
        """
        Args:
            x (torch.Tensor): audio batch
            video_emb (torch.Tensor): video embeddings batch
        """
        raise NotImplementedError()


class RTFSNet2SpeakersSeparation(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.rtfs_net = RTFSNet(*args, **kwargs)

    def forward(
        self, mix: torch.Tensor, s1_video_emb: torch.Tensor, s2_video_emb: torch.Tensor
    ):
        """
        Args:
            x (torch.Tensor): audio batch
            s1_video_emb (torch.Tensor): first speaker video embeddings batch
            s2_video_emb (torch.Tensor): second speaker video embeddings batch
        """
        s1 = self.rtfs_net(mix, s1_video_emb)
        s2 = self.rtfs_net(mix, s2_video_emb)

        preds = torch.cat([s1, s2], dim=1)
        return {"preds": preds}


class RTFSNetVideoEncoding2SpeakersSeparation(
    nn.Module
):  # this version of the model applies video encoder
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.rtfs_net = RTFSNet2SpeakersSeparation(*args, **kwargs)

    def forward(
        self, mix: torch.Tensor, s1_mouth: torch.Tensor, s2_mouth: torch.Tensor
    ):
        """
        Args:
            x (torch.Tensor): audio batch
            s1_video_emb (torch.Tensor): first speaker video embeddings batch
            s2_video_emb (torch.Tensor): second speaker video embeddings batch
        """
        # s1_video_emb = video_enc(s1_mouth)
        # s2_video_emb = video_enc(s2_mouth)
        # s1 = self.rtfs_net(mix, s1_video_emb)
        # s2 = self.rtfs_net(mix, s2_video_emb)

        # preds = torch.cat([s1, s2], dim=1)
        # return {'preds': preds} TODO

        raise NotImplementedError()
