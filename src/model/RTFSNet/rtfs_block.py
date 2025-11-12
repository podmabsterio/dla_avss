import torch
from torch import nn


class RTFSBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(a: torch.Tensor):
        """
        Args:
            a (torch.Tensor): tensor A from paper with shape (B, C_a, T_a, F)
        """
        raise NotImplementedError()
