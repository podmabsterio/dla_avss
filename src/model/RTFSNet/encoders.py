import torch
from torch import nn


class VideoEncoder(nn.Module):
    def __init__(self, video_encoder_model_name):
        super().__init__()

    def forward(y: torch.Tensor):
        """
        Args:
            y (torch.Tensor): video batch
        Outputs:
            v (torch.Tensor): encoded video with shape (B, C_v, T_v)
        """
        raise NotImplementedError()


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): audio batch
        Outputs:
            a_0 (torch.Tensor): encoded video with shape (B, C_a, T_a, F)
        """
        raise NotImplementedError()
