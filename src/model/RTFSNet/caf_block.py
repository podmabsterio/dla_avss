import torch
from torch import nn


class CAFBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(a: torch.Tensor):
        """
        Args:
            a_1 (torch.Tensor): processed encoded audio batch with shape (B, C_a, T_a, F)
            v_1 (torch.Tensor): processed encoded audio batch with shape (B, C_v, T_v)
        Outputs:
            a_2 (torch.Tensor): CAF processed audio-video tensor with shape (B, C_a, T_a, F)
        """
        raise NotImplementedError()
