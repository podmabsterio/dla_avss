import torch
from torch import nn


class VPBlock(nn.Module):
    def __init__(self, video_encoder_model_name):
        super().__init__()

    def forward(y: torch.Tensor):
        """
        Args:
            v_0 (torch.Tensor): encoded video with shape (B, C_v, T_v)
        Outputs:
            v_1 (torch.Tensor): processed video with shape (B, C_v, T_v)
        """
        raise NotImplementedError()