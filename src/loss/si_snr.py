from torch import nn
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class SISNRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.metric = ScaleInvariantSignalNoiseRatio()

    def forward(self, preds, target, **batch):
        loss = -self.metric(preds, target)
        return {"loss": loss}
