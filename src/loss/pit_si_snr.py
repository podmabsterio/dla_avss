from torch.nn import Module
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio


class PITSISNRLoss(Module):
    def __init__(self):
        super().__init__()
        self.objective = PermutationInvariantTraining(
            scale_invariant_signal_noise_ratio, mode="speaker-wise", eval_func="max"
        )

    def forward(self, preds, target):
        return -self.objective(preds, target)
