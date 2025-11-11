from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def __init__(self, *args, examples_to_log_on_val=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.examples_to_log_on_val = examples_to_log_on_val

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_only_mix(**batch)
        else:
            # Log Stuff
            print("examples_to_log_on_val:", self.examples_to_log_on_val)
            self.log_all_audios(examples_to_log=self.examples_to_log_on_val, **batch)

    def log_only_mix(self, mix, **batch):
        self.log_audio(mix[0], "mix")

    def log_all_audios(self, mix, preds, target, examples_to_log=5, **batch):
        for i in range(examples_to_log):
            self.log_audio(mix[i], f"mix_{i + 1}")
            self.log_audio(preds[i, 0, :].unsqueeze(0), f"pred1_{i + 1}")
            self.log_audio(preds[i, 1, :].unsqueeze(0), f"pred2_{i + 1}")
            self.log_audio(target[i, 0, :].unsqueeze(0), f"target1_{i + 1}")
            self.log_audio(target[i, 1, :].unsqueeze(0), f"target2_{i + 1}")

    def log_audio(self, audio, audio_name):
        audio_for_writer = audio.detach().cpu()
        sample_rate = self.sample_rate_for_logging
        self.writer.add_audio(audio_name, audio_for_writer, sample_rate=sample_rate)
