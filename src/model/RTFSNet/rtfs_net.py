import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(x: torch.Tensor, y: torch.Tensor):
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
