import torch
import torch.nn.functional as FF
from sru import SRU
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


class ChannelsFeatsNormalization(nn.Module):
    def __init__(self, num_channels, num_feats):
        super().__init__()
        self.ln = nn.LayerNorm((num_channels, num_feats))

    def forward(self, x: torch.Tensor):
        result = x.permute(0, 2, 1, 3)
        result = self.ln(result)
        result = result.permute(0, 2, 1, 3)
        return result


class TFMHSAProjection(nn.Module):
    def __init__(self, in_channels, num_feats, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.prelu = nn.PReLU()
        self.normalization = ChannelsFeatsNormalization(out_channels, num_feats)

    def forward(self, x):
        result = self.conv(x)
        result = self.prelu(result)
        result = self.normalization(result)
        return result


class TFDomainSelfAttention(nn.Module):
    def __init__(self, in_channels, num_feats, num_heads, hidden_channels):
        super().__init__()

        assert in_channels % num_heads == 0

        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

        self.qs = nn.ModuleList(
            [
                TFMHSAProjection(in_channels, num_feats, hidden_channels)
                for _ in range(num_heads)
            ]
        )
        self.ks = nn.ModuleList(
            [
                TFMHSAProjection(in_channels, num_feats, hidden_channels)
                for _ in range(num_heads)
            ]
        )
        self.vs = nn.ModuleList(
            [
                TFMHSAProjection(in_channels, num_feats, in_channels // num_heads)
                for _ in range(num_heads)
            ]
        )

        self.out_proj = TFMHSAProjection(in_channels, num_feats, in_channels)

    def forward(self, x: torch.Tensor):
        B, C, T, F = x.shape
        residual = x

        all_Q = [q(x) for q in self.qs]
        all_K = [k(x) for k in self.ks]
        all_V = [v(x) for v in self.vs]

        Q = torch.cat(all_Q, dim=0)
        K = torch.cat(all_K, dim=0)
        V = torch.cat(all_V, dim=0)
        B_head = B * self.num_heads

        Q = Q.transpose(1, 2).contiguous()
        Q = Q.view(B_head, T, self.hidden_channels * F)

        K = K.transpose(1, 2).contiguous()
        K = K.view(B_head, T, self.hidden_channels * F)

        V = V.transpose(1, 2).contiguous()
        old_shape = V.shape
        V = V.view(B_head, T, -1)

        Q_sd = Q.transpose(0, 1)
        K_sd = K.transpose(0, 1)
        V_sd = V.transpose(0, 1)

        attn_out = FF.scaled_dot_product_attention(Q_sd, K_sd, V_sd)

        V = attn_out.transpose(0, 1).contiguous()

        V = V.view(old_shape)
        V = V.transpose(1, 2).contiguous()
        emb_dim = V.shape[1]

        x = V.view(self.num_heads, B, emb_dim, T, F)
        x = x.transpose(0, 1).contiguous()
        x = x.view(B, self.num_heads * emb_dim, T, F)

        x = self.out_proj(x)
        return x + residual


class DualPathBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        num_rnn_layers,
        dimension,
        kernel_size,
        use_sru=False,
    ):
        super().__init__()
        self.is_feature_dimension = None
        if dimension == "T":
            self.is_feature_dimension = False
        elif dimension == "F":
            self.is_feature_dimension = True
        else:
            raise ValueError("dimension can be either 'T' or 'F'")

        self.ln = nn.LayerNorm(in_channels)
        if use_sru:
            self.rnn = SRU(
                in_channels * kernel_size,
                hidden_dim,
                num_rnn_layers,
                bidirectional=True,
            )
        else:
            self.rnn = nn.LSTM(
                in_channels * kernel_size,
                hidden_dim,
                num_rnn_layers,
                bidirectional=True,
            )

        self.conv_transpose = nn.ConvTranspose1d(
            hidden_dim * 2, in_channels, kernel_size
        )
        self.kernel_size = kernel_size

    def forward(self, a: torch.Tensor):
        if self.is_feature_dimension:
            a = a.transpose(2, 3).contiguous()
        B, C, T, F = a.shape  # a: (B, C, T, F)
        residual = a

        a = a.permute(0, 2, 3, 1).contiguous()  # (B, T, F, C)
        a = self.ln(a)
        a = a.permute(0, 2, 3, 1).contiguous()  # (B, F, C, T)
        a = FF.unfold(
            a.view(B * F, C, T, 1), (self.kernel_size, 1)
        )  # (B * F, C * kernel_size, T_new)
        a = a.permute(2, 0, 1).contiguous()  # (T_new, B * F, C * kernel_size)
        a = self.rnn(a)[0]
        a = a.permute(1, 2, 0).contiguous()  # (B * F, C * kernel_size, T)
        a = self.conv_transpose(a)  # (B * F, C, T)
        a = a.view([B, F, C, T])
        a = a.permute(0, 2, 3, 1).contiguous()  # (B, C, T, F)
        a = a + residual

        if self.is_feature_dimension:
            a = a.transpose(2, 3).contiguous()

        return a


class CompressionModule(nn.Module):
    def __init__(self, in_channels, inner_channels, q):
        super().__init__()

        self.downsampling_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=inner_channels, kernel_size=1
        )
        self.gln = nn.GroupNorm(num_groups=1, num_channels=inner_channels)
        self.prelu = nn.PReLU()

        self.compression_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
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
        F_out = _conv_chain_out_size(a.shape[-1], q)
        downsampled = a
        compressed = 0
        downsampled_layers = []
        for i in range(len(self.compression_convs)):
            downsampled = self.compression_convs[i](downsampled)
            downsampled_layers.append(downsampled)
            compressed += FF.adaptive_avg_pool2d(downsampled, (T_out, F_out))

        return compressed, downsampled_layers


class TemporalFrequencyAttentionRecognition(nn.Module):
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

        self.w1 = nn.Conv2d(**conv_params)
        self.w1_norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)

        self.w2 = nn.Conv2d(**conv_params)
        self.w2_norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)

        self.w3 = nn.Conv2d(**conv_params)
        self.w3_norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, m, n):
        interpolation_target = (m.shape[2], m.shape[3])

        mul1 = self.sigmoid(self.w1_norm(self.w1(n)))
        mul1 = FF.interpolate(mul1, size=interpolation_target, mode="nearest")

        mul2 = self.w2_norm(self.w2(m))

        add = FF.interpolate(
            self.w3_norm(self.w3(n)), size=interpolation_target, mode="nearest"
        )

        return mul1 * mul2 + add


class ReconstructionModule(nn.Module):
    def __init__(self, in_channels, q):
        super().__init__()
        self.tf_ar_fusion_layers = nn.ModuleList(
            [TemporalFrequencyAttentionRecognition(in_channels) for _ in range(q)]
        )
        self.tf_ar_residual_layers = nn.ModuleList(
            [TemporalFrequencyAttentionRecognition(in_channels) for _ in range(q - 1)]
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


class RTFSBlock(nn.Module):
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
        self.scaling_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            groups=in_channels,
        )
        self.prelu = nn.PReLU()
        self.downsampling = CompressionModule(in_channels, hidden_channels, q)

        self.dual_path1 = DualPathBlock(
            hidden_channels,
            rnn_hidden_channels,
            num_rnn_layers,
            "F",
            rnn_kernel_size,
            use_sru,
        )
        self.dual_path2 = DualPathBlock(
            hidden_channels,
            rnn_hidden_channels,
            num_rnn_layers,
            "T",
            rnn_kernel_size,
            use_sru,
        )

        compressed_num_feats = _conv_chain_out_size(num_feats, q)
        self.attn = TFDomainSelfAttention(
            hidden_channels, compressed_num_feats, num_heads, attn_hidden_channels
        )

        self.upsampling = ReconstructionModule(hidden_channels, q)
        self.upsampling_conv = nn.Conv2d(hidden_channels, in_channels, 1)

    def forward(self, a: torch.Tensor):
        """
        Args:
            a_2 (torch.Tensor): tensor A from paper with shape (B, C_a, T_a, F)
        Outputs:
            a_R (torch.Tensor): tensor A'' from paper with shape (B, C_a, T_a, F)
        """
        residual = self.prelu(self.scaling_conv(a))
        a, downsampled_layers = self.downsampling(residual)
        a = self.dual_path1(a)
        a = self.dual_path2(a)
        a = self.attn(a)
        a = self.upsampling(downsampled_layers, a)
        a = self.upsampling_conv(a)

        return a + residual
