import torch
from torchmetrics.audio import PermutationInvariantTraining as PIT
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio as si_sdr,
)
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from torchmetrics.functional.audio.pesq import (
    perceptual_evaluation_speech_quality as pesq,
)
from torchmetrics.functional.audio.stoi import (
    short_time_objective_intelligibility as stoi,
)

from src.metrics.base_metric import BaseMetric


class PITMetric(BaseMetric):
    def __init__(self, metric, name, device, metric_params=None, eval_func="max"):
        super().__init__(name)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if metric_params is None:
            metric_params = {}

        self.metric = PIT(
            metric, mode="speaker-wise", eval_func=eval_func, **metric_params
        ).to(device)

    def __call__(self, preds: torch.Tensor, target: torch.Tensor, **batch):
        return self.metric(preds, target)


class PIT_SISNR(PITMetric):
    def __init__(self, name, device):
        super().__init__(si_snr, name, device)


class PIT_STOI(PITMetric):
    def __init__(self, sample_rate, name, device):
        params = dict(fs=sample_rate)
        super().__init__(stoi, name, device, params)


class PIT_PESQ(PITMetric):
    def __init__(self, sample_rate, name, device, mode, n_processes):
        def pesq_wrapper(preds, target):
            pesq(
                preds, target, sample_rate, mode, n_processes=n_processes
            )  # because of collision of named params "mode" of PIT and pesq

        super().__init__(pesq_wrapper, name, device)


class PIT_SISNRi(BaseMetric):
    def __init__(self, name, device):
        super().__init__(name)
        self.pit_sisnr = PIT_SISNR(name, device)

    def __call__(
        self, mix: torch.Tensor, preds: torch.Tensor, target: torch.Tensor, **batch
    ):
        return (
            self.pit_sisnr(preds, target) - si_snr(mix.expand(-1, 2, -1), target).mean()
        )
