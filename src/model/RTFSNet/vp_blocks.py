import math

import torch
import torch.nn.functional as FF
from torch import nn


def _conv_chain_out_size(dim_size, q):
    for i in range(q - 1):
        if dim_size < 4:
            raise ValueError(
                f"Insufficient tensor size to apply convolution on step {i + 2}"
            )
        else:
            dim_size = (dim_size - 2) // 2 + 1

    return dim_size


# class SinusoidalPositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, max_len: int = 10000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model, requires_grad=False)
#         position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2, dtype=torch.float32)
#             * (-math.log(10000.0) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer("pe", pe, persistent=False)

#     def forward(self, x):
#         T = x.size(1)
#         return x + self.pe[:T].unsqueeze(0)


# class MultiHeadSelfAttention(nn.Module):
#     def __init__(
#         self,
#         in_chan: int,
#         n_head: int = 8,
#         dropout: int = 0.1,
#         positional_encoding: bool = True,
#         batch_first=True,
#         *args,
#         **kwargs,
#     ):
#         super(MultiHeadSelfAttention, self).__init__()
#         self.in_chan = in_chan
#         self.n_head = n_head
#         self.dropout = dropout
#         self.positional_encoding = positional_encoding
#         self.batch_first = batch_first

#         assert (
#             self.in_chan % self.n_head == 0
#         ), "In channels: {} must be divisible by the number of heads: {}".format(
#             self.in_chan, self.n_head
#         )

#         self.norm1 = nn.LayerNorm(self.in_chan)
#         self.pos_enc = (
#             PositionalEncoding(self.in_chan)
#             if self.positional_encoding
#             else nn.Identity()
#         )
#         self.attention = nn.MultiheadAttention(
#             self.in_chan, self.n_head, self.dropout, batch_first=self.batch_first
#         )
#         self.dropout_layer = nn.Dropout(self.dropout)
#         self.norm2 = nn.LayerNorm(self.in_chan)
#         self.drop_path_layer = DropPath(self.dropout)

#     def forward(self, x: torch.Tensor):
#         res = x
#         if self.batch_first:
#             x = x.transpose(1, 2)  # B, C, T -> B, T, C

#         x = self.norm1(x)
#         x = self.pos_enc(x)
#         residual = x
#         x = self.attention(x, x, x)[0]
#         x = self.dropout_layer(x) + residual
#         x = self.norm2(x)

#         if self.batch_first:
#             x = x.transpose(2, 1)  # B, T, C -> B, C, T

#         x = self.drop_path_layer(x) + res
#         return x


# class GlobalAttention(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         hidden_channels,
#         kernel_size,
#         num_head,
#         dropout,
#     ):
#         super(GlobalAttention, self).__init__()

#         assert self.in_chan % self.n_head

#         self.ln1 = nn.LayerNorm(in_channels)
#         self.pos_enc = (
#             SinusoidalPositionalEncoding(self.in_chan)
#             if self.positional_encoding
#             else nn.Identity()
#         )
#         self.attention = nn.MultiheadAttention(
#             self.in_chan, self.n_head, self.dropout, batch_first=self.batch_first
#         )
#         self.dropout_layer = nn.Dropout(self.dropout)
#         self.norm2 = nn.LayerNorm(self.in_chan)
#         self.drop_path_layer = DropPath(self.dropout)

#         self.MHSA = nn.MultiHeadSelfAttention(
#             self.in_chan, self.n_head, self.dropout, self.pos_enc
#         )
#         self.FFN = conv_layers.get(self.ffn_name)(
#             self.in_chan, self.hid_chan, self.kernel_size, dropout=self.dropout
#         )

#     def forward(self, x: torch.Tensor):
#         x = self.MHSA(x)
#         x = self.FFN(x)
#         return x


class CompressionModule1D(nn.Module):
    def __init__(self, in_channels, inner_channels, q):
        super().__init__()

        self.downsampling_conv = nn.Conv1d(
            in_channels=in_channels, out_channels=inner_channels, kernel_size=1
        )
        self.gln = nn.GroupNorm(num_groups=1, num_channels=inner_channels)
        self.prelu = nn.PReLU()

        self.compression_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=inner_channels,
                        out_channels=inner_channels,
                        kernel_size=4,
                        stride=1 if i == 0 else 2,
                        groups=inner_channels,
                        padding="same" if i == 0 else 1,
                    ),
                    nn.GroupNorm(num_groups=1, num_channels=inner_channels),
                )
                for i in range(q)
            ]
        )

    def forward(self, a):
        a = self.downsampling_conv(a)
        a = self.gln(a)
        a = self.prelu(a)
        q = len(self.compression_convs)

        T_out = _conv_chain_out_size(a.shape[-2], q)
        downsampled = a
        compressed = 0
        downsampled_layers = []
        for i in range(len(self.compression_convs)):
            downsampled = self.compression_convs[i](downsampled)
            downsampled_layers.append(downsampled)
            compressed += FF.adaptive_avg_pool1d(downsampled, (T_out))

        return compressed, downsampled_layers


class TemporalFrequencyAttentionRecognition1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        conv_params = dict(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=4,
            groups=in_channels,
            bias=False,
            padding="same",
        )

        self.w1 = nn.Conv1d(**conv_params)
        self.w1_norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)

        self.w2 = nn.Conv1d(**conv_params)
        self.w2_norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)

        self.w3 = nn.Conv1d(**conv_params)
        self.w3_norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, m, n):
        interpolation_target = m.shape[2]

        mul1 = self.sigmoid(self.w1_norm(self.w1(n)))
        mul1 = FF.interpolate(mul1, size=interpolation_target, mode="nearest")

        mul2 = self.w2_norm(self.w2(m))

        add = FF.interpolate(
            self.w3_norm(self.w3(n)), size=interpolation_target, mode="nearest"
        )

        return mul1 * mul2 + add


class ReconstructionModule1D(nn.Module):
    def __init__(self, in_channels, q):
        super().__init__()
        self.tf_ar_fusion_layers = nn.ModuleList(
            [TemporalFrequencyAttentionRecognition1D(in_channels) for _ in range(q)]
        )
        self.tf_ar_residual_layers = nn.ModuleList(
            [TemporalFrequencyAttentionRecognition1D(in_channels) for _ in range(q - 1)]
        )

    def forward(self, downsampled, embedding):
        q = len(self.tf_ar_fusion_layers)
        fused = [
            self.tf_ar_fusion_layers[i](downsampled[i], embedding) for i in range(q)
        ]

        upsampled = (
            self.tf_ar_residual_layers[-1](fused[-2], fused[-1]) + downsampled[-2]
        )
        for i in range(q - 3, -1, -1):
            upsampled = (
                self.tf_ar_residual_layers[i](fused[i], upsampled) + downsampled[i]
            )

        return upsampled


class VPBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        rnn_hidden_channels,
        q,
        num_rnn_layers,
        rnn_kernel_size,
        num_feats,
        num_heads,
        attn_hidden_channels,
        use_sru=False,
    ):
        super().__init__()
        self.scaling_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            groups=in_channels,
        )
        self.prelu = nn.PReLU()
        self.downsampling = CompressionModule1D(in_channels, hidden_channels, q)

        self.upsampling = ReconstructionModule1D(hidden_channels, q)
        self.upsampling_conv = nn.Conv1d(hidden_channels, in_channels, 1)

    def forward(self, a: torch.Tensor):
        """
        Args:
            a_2 (torch.Tensor): tensor A from paper with shape (B, C_a, T_a)
        Outputs:
            a_R (torch.Tensor): tensor A'' from paper with shape (B, C_a, T_a)
        """
        residual = self.prelu(self.scaling_conv(a))
        a, downsampled_layers = self.downsampling(residual)
        a = self.dual_path1(a)
        a = self.dual_path2(a)
        a = self.attn(a)
        a = self.upsampling(downsampled_layers, a)
        a = self.upsampling_conv(a)

        return a + residual
