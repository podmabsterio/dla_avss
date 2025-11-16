import torch
import torch.nn as nn
from torch import Tensor


class S3Block(nn.Module):
    def __init__(self, c_a: int):
        """
        Args:
            c_a: C_a из статьи — число каналов у a_0 и a_R.
                 Должно быть чётным: половина под real, половина под imag.
        """
        super().__init__()
        assert c_a % 2 == 0, "C_a must be even (half real, half imag)"

        self.c_a = c_a
        self.trunk = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(
                in_channels=self.c_a,
                out_channels=self.c_a,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU()
        )

    def forward(self, a_0: Tensor, a_r: Tensor) -> Tensor:
        """
        Args:
            a_0: тензор a_0 из статьи, форма (B, C_a, T_a, F)
            a_r: тензор a_R из статьи, форма (B, C_a, T_a, F)
        Returns:
            z:  z из статьи, форма (B, C_a, T_a, F)
        """
        m = self.trunk(a_r)

        m_r, m_i = torch.chunk(m, 2, dim=1)
        E_r, E_i = torch.chunk(a_0, 2, dim=1)

        z_r = m_r * E_r - m_i * E_i
        z_i = m_r * E_i + m_i * E_r

        z = torch.cat([z_r, z_i], dim=1)

        return z
