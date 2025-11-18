import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, in_chan, hop_length, win):
        super().__init__()

        self.win = win
        self.hop_length = hop_length
        self.in_chan = in_chan

        self.decoder = nn.ConvTranspose2d(
            in_channels=in_chan, out_channels=2, kernel_size=3, padding=1
        )
        torch.nn.init.xavier_uniform_(self.decoder.weight)

        self.register_buffer("window", torch.hann_window(win), False)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z (torch.Tensor): tensor a_R from paper with shape (B, C_a, T_a, F)
        Outputs:
            separated_audio (torch.Tensor): separated audio batch
        """
        batch_size = z.shape[0]

        z = z.view(batch_size, self.in_chan, *z.shape[-2:])

        decoded = self.decoder(z)
        spec = torch.complex(decoded[:, 0], decoded[:, 1])
        spec = spec.transpose(1, 2).contiguous()

        audio = torch.istft(
            spec,
            n_fft=self.win,
            hop_length=self.hop_length,
            window=self.window.to(z.device),
        )

        return audio.view(batch_size, 1, -1)
