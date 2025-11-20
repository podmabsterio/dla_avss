import torch
from torch import nn


class ComplexSpecMappingMSELoss(nn.Module):
    def __init__(self, win=256, hop=128):
        super().__init__()
        self.win = win
        self.hop = hop
        self.register_buffer("window", torch.hann_window(self.win), False)

    def forward(self, pred_specs: torch.Tensor, target: torch.Tensor, **batch):
        target_spec = torch.stft(
            target.squeeze(1),
            n_fft=self.win,
            hop_length=self.hop,
            window=self.window.to(target.device),
            return_complex=True,
        )

        diff = pred_specs - target_spec
        loss = (diff.real**2 + diff.imag**2).mean()
        return {"loss": loss}
