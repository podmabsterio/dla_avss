import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import move_batch_transforms_to_device, transform_batch
from src.datasets.eval_bss_dataset import EvalBssDataset
from src.metrics.tracker import MetricTracker
from src.utils.init_utils import set_random_seed, set_worker_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="calc_metrics")
def main(config):
    set_random_seed(config.metric_calculator.seed)

    if config.metric_calculator.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.metric_calculator.device

    batch_transforms = instantiate(config.transforms.batch_transforms)
    instance_transforms = instantiate(config.transforms.instance_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    dataset = EvalBssDataset(
        config.metric_calculator.gt_path,
        config.metric_calculator.pred_path,
        instance_transforms=instance_transforms["inference"],
    )
    dataloader = instantiate(
        config.dataloader,
        dataset=dataset,
        drop_last=False,
        shuffle=False,
        worker_init_fn=set_worker_seed,
    )

    metrics = instantiate(config.metrics)
    assert len(metrics) > 0, "Please provide at least one metric"
    evaluation_metrics = MetricTracker(
        *[m.name for m in metrics["inference"]],
        writer=None,
    )

    for batch in dataloader:
        batch = transform_batch(batch_transforms, batch, "inference")
        for met in metrics["inference"]:
            evaluation_metrics.update(met.name, met(**batch))

    logs = evaluation_metrics.result()
    for key, value in logs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
