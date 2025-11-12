import torch
from torch import nn


class S3Block(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(a_0, a_r: torch.Tensor):
        """
        Args:
            a_0 (torch.Tensor): tensor a_0 from paper with shape (B, C_a, T_a, F)
            a_r (torch.Tensor): tensor a_R from paper with shape (B, C_a, T_a, F)
        Outputs:
            z (torch.Tensor): tensor a_R from paper with shape (B, C_a, T_a, F)
        """
        raise NotImplementedError()
