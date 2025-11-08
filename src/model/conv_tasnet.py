import torch
from torch import nn
from src.model.base_model import BaseModel
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size,
        padding,
        dilation=1
    ):
        super().__init__()
        self.common_part = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=1
            ),
            nn.PReLU(),
            nn.GroupNorm(
                num_groups=1,
                num_channels=hidden_channels
            ),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                groups=hidden_channels,
                padding=padding
            ),
            nn.PReLU(),
            nn.GroupNorm(
                num_groups=1,
                num_channels=hidden_channels
            )
        )

        self.res_out = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=input_channels,
            kernel_size=1
        )
        
        self.skip_out = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=input_channels,
            kernel_size=1
        )

    def forward(self, input):
        common_out = self.common_part(input)
        residual = self.res_out(common_out)
        skip = self.skip_out(common_out)
        return residual, skip
        
class TCN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        bottleneck_dim,
        hidden_dim,
        num_layers,
        num_stacks,
        kernel_size=3,
    ):
        super().__init__()
        
        self.layer_norm = nn.GroupNorm(
            num_groups=1,
            num_channels=input_dim
        )
        self.bottleneck = nn.Conv1d(
            in_channels=input_dim,
            out_channels=bottleneck_dim,
            kernel_size=1
        )
        
        self.tcn = nn.ModuleList([])
        for _ in range(num_stacks):
            for i in range(num_layers):
                self.tcn.append(
                    ConvBlock1D(
                        input_channels=bottleneck_dim,
                        hidden_channels=hidden_dim,
                        kernel_size=kernel_size,
                        dilation=2**i,
                        padding=2**i
                    )
                ) 
                    
        self.output = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(
                in_channels=bottleneck_dim,
                out_channels=output_dim,
                kernel_size=1
            )
        )
        
    def forward(self, input):
        output = self.bottleneck(self.layer_norm(input))

        skip_connection = None
        for block in self.tcn:
            residual, skip = block(output)
            output = output + residual
            skip_connection = skip_connection + skip if skip_connection is not None else skip
            
        output = self.output(skip_connection)
        return output

class ConvTasNet(BaseModel):
    def __init__(
        self,
        encoder_dim=512,
        sample_rate=16000,
        slice_factor=2,
        tcn_bottleneck_dim=128,
        tcn_num_layers=8,
        tcn_num_stacks=3,
        tcn_kernel_size=3,
        num_speakers=2
    ):
        super().__init__()

        self.slice_len = int((sample_rate * slice_factor) // 1000)
        self.stride = self.slice_len // 2
        self.encoder_dim = encoder_dim
        self.num_speakers = num_speakers

        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=encoder_dim,
            kernel_size=self.slice_len,
            stride=self.stride
        )

        self.tcn_module = TCN(
            input_dim=encoder_dim,
            output_dim=encoder_dim * num_speakers,
            bottleneck_dim=tcn_bottleneck_dim,
            hidden_dim=tcn_bottleneck_dim * 4,
            num_layers=tcn_num_layers,
            num_stacks=tcn_num_stacks,
            kernel_size=tcn_kernel_size
        )

        self.decoder = nn.ConvTranspose1d(
            in_channels=encoder_dim,
            out_channels=1,
            kernel_size=self.slice_len,
            stride=self.stride
        )

    def pad_signal(self, x):
        _, _, T = x.size()
        rest = (self.slice_len - (T % self.slice_len)) % self.slice_len
        if rest > 0:
            x = F.pad(x, (0, rest), mode='constant')
        return x, rest

    def forward(self, mix, **batch):
        """
        Forward pass of Conv-TasNet.

        Args:
            mix (torch.Tensor): 
                Input mixture waveform of shape (B, 1, T),
                where B is the batch size and T is the number of samples.

        Returns:
            torch.Tensor:
                Separated waveforms of shape (B, C, T),
                where C is the number of estimated sources.
        """

        mix, rest = self.pad_signal(mix)
        B = mix.size(0)

        enc = self.encoder(mix)  # (B, N, L)

        masks_logits = self.tcn_module(enc)  # (B, N*C, L)
        masks = torch.sigmoid(masks_logits).view(B, self.num_speakers, self.encoder_dim, -1)  # (B, C, N, L)

        masked = enc.unsqueeze(1) * masks  # (B, C, N, L)
        dec_in = masked.view(B * self.num_speakers, self.encoder_dim, -1)  # (B*C, N, L)
        preds = self.decoder(dec_in)  # (B*C, 1, T')

        if rest > 0:
            preds = preds[:, :, :-rest]

        preds = preds.view(B, self.num_speakers, -1)  # (B, C, T)

        return {"preds": preds}
