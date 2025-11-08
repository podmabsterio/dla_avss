import torch
from torchmetrics.audio import PermutationInvariantTraining as PIT
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from torchmetrics.functional.audio.pesq import (
    perceptual_evaluation_speech_quality as pesq,
)
from torchmetrics.functional.audio.stoi import (
    short_time_objective_intelligibility as stoi,
)

from src.metrics.base_metric import BaseMetric


class PITMetric(BaseMetric):
    def __init__(
        self, metric, device, eval_func="max", *args, metric_params=None, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if metric_params is None:
            metric_params = {}

        self.metric = PIT(
            metric, mode="speaker-wise", eval_func=eval_func, **metric_params
        ).to(device)

    def __call__(self, preds: torch.Tensor, target: torch.Tensor, **kwargs):
        return self.metric(preds, target)


class PIT_SISNR(PITMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(si_snr, *args, **kwargs)
        
 
class PIT_SISNRi(PITMetric):
    def __init__(self, *args, **kwargs):
        si_snri = lambda mix, preds, target: si_snr(preds, target) - si_snr(mix, target)
        super().__init__(si_snri, *args, **kwargs)


class PIT_STOI(PITMetric):
    def __init__(self, sample_rate, *args, **kwargs):
        params = dict(fs=sample_rate)
        super().__init__(stoi, *args, metric_params=params, **kwargs)


class PIT_PESQ(PITMetric):
    def __init__(self, sample_rate, *args, **kwargs):
        params = dict(fs=sample_rate)
        super().__init__(pesq, *args, metric_params=params, **kwargs)
