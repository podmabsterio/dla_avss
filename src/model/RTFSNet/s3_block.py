import torch
from torch import nn


class S3Block(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(y: torch.Tensor):
        """
        Args:
            y (torch.Tensor): video batch
        Outputs:
            v (torch.Tensor): encoded video with shape (B, C_v, T_v)
        """
        raise NotImplementedError()
