import torch
from torch import nn


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

    def forward(x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x (torch.Tensor): audio batch
            y (torch.Tensor): video batch
        """
        raise NotImplementedError()
