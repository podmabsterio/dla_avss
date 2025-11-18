from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn


class CAFBlock(nn.Module):
    def __init__(self, chan_audio: int, chan_video: int):
        super().__init__()
        self.chan_audio = chan_audio
        self.chan_video = chan_video

        self.gate_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.chan_audio,
                out_channels=self.chan_audio,
                kernel_size=1,
                groups=self.chan_audio,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=chan_audio),
            nn.ReLU(),
        )

        self.value_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.chan_audio,
                out_channels=self.chan_audio,
                kernel_size=1,
                groups=self.chan_audio,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=chan_audio),
        )

        self.attention_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.chan_video,
                out_channels=self.chan_audio,
                kernel_size=1,
                groups=self.chan_audio,
            ),
            nn.GroupNorm(
                num_groups=1,
                num_channels=self.chan_audio,
            ),
        )

        self.resize_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.chan_video,
                out_channels=self.chan_audio,
                kernel_size=1,
                groups=self.chan_audio,
            ),
            nn.GroupNorm(
                num_groups=1,
                num_channels=self.chan_audio,
            ),
        )

    def forward(self, audio: torch.Tensor, video_emb: torch.Tensor):
        """
        Args:
            audio (torch.Tensor):
                Audio feature map of shape (B, C_a, T, F_a), where:
                    B  — batch size
                    C_a — number of audio channels (chan_audio)
                    T  — temporal dimension
                    F_a — frequency or spatial feature dimension

            video_emb (torch.Tensor):
                Video feature map of shape (B, C_v, T_v), where:
                    C_v — number of video channels (chan_video)
                    T_v — video time

        Returns:
            torch.Tensor:
                Fused audio–video representation of shape (B, C_a, T, F_a).
                Computed as f1 + f2, where:
                    f1 — audio values scaled by softmax attention derived from video,
                    f2 — gated audio features modulated by temporally-aligned video embeddings.
        """
        batch_size, _, time_steps, _ = audio.shape

        a_v = self.value_conv(audio)
        att = self.attention_conv(video_emb)
        att = att.reshape(batch_size, self.chan_audio, 1, -1)
        att = att.mean(2, keepdim=False).view(batch_size, self.chan_audio, -1)
        att = F.interpolate(torch.softmax(att, -1), size=time_steps, mode="nearest")
        f1 = att.unsqueeze(-1) * a_v

        video_resized = self.resize_conv(video_emb)
        video_interpolated = F.interpolate(
            video_resized, size=time_steps, mode="nearest"
        )
        video_interpolated = video_interpolated.unsqueeze(-1)
        a_gate = self.gate_conv(audio)
        f2 = a_gate * video_interpolated

        return f1 + f2
