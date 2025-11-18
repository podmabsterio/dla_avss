from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.model.RTFSNet.resnet import ResNet


def _3D_to_2D_tensor(x: torch.Tensor) -> torch.Tensor:
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)


class Lipreading(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend_nout = 64
        self.backend_out = 512

        self.trunk = ResNet([2, 2, 2, 2])

        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            nn.SiLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, T, H, W] → [B, T_new, 512]
        """
        b = x.shape[0]
        x = self.frontend3D(x)
        t_new = x.shape[2]
        x = _3D_to_2D_tensor(x)
        x = self.trunk(x)
        return x.view(b, t_new, x.size(1))


class VideoEncoder(nn.Module):
    def __init__(
        self,
        weights_folder,
        video_encoder_model_name: str = "lrw_resnet18_dctcn_video_boundary",
        device: str = "cpu",
    ):
        super().__init__()
        if video_encoder_model_name.lower() != "lrw_resnet18_dctcn_video_boundary":
            raise ValueError(f"Unknown model: {video_encoder_model_name}")
        self.encoder = Lipreading()

        weights_path = Path(weights_folder) / f"{video_encoder_model_name}.pth"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Cannot find weights file: {weights_path.resolve()}"
            )

        state = torch.load(weights_path, map_location=device)

        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.encoder.load_state_dict(state, strict=False)

        self.to(device)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y (torch.Tensor): video batch [B, 1, T, H, W]
        Returns:
            v (torch.Tensor): encoded video with shape (B, C_v, T_v)
        """
        v = self.encoder(y)
        return v.permute(0, 2, 1).contiguous()


class AudioEncoder(nn.Module):
    def __init__(
        self,
        win: int,
        hop_length: int,
        out_chan: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()

        self.win = win
        self.hop_length = hop_length
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=self.out_chan,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.GroupNorm(
                num_groups=1,
                num_channels=self.out_chan,
            ),
        )
        nn.init.xavier_uniform_(self.conv[0].weight)

        self.register_buffer("window", torch.hann_window(self.win), False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): audio batch (B, 1, T_a)
        Outputs:
            a_0 (torch.Tensor): encoded audio with shape (B, C_a, T_a, F)
        """
        x = x.squeeze(1)

        spec = torch.stft(
            x,
            n_fft=self.win,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            return_complex=True,
        )

        spec = torch.stack([spec.real, spec.imag], 1).transpose(2, 3).contiguous()
        spec_feature_map = self.conv(spec)

        return spec_feature_map
