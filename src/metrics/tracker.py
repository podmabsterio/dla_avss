class MetricTracker:
    """
    Class to aggregate metrics from many batches (without pandas).
    """

    def __init__(self, *keys, writer=None):
        """
        Args:
            *keys (list[str]): metric names (e.g. "loss", "accuracy").
            writer (optional): experiment tracker (e.g. WandB, Comet).
        """
        self.writer = writer
        self._data = {key: {"total": 0.0, "counts": 0, "average": 0.0} for key in keys}

    def reset(self):
        """
        Reset all stored metric values.
        """
        for key in self._data:
            self._data[key]["total"] = 0.0
            self._data[key]["counts"] = 0
            self._data[key]["average"] = 0.0

    def update(self, key, value, n=1):
        """
        Update metric with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): number of samples for this batch.
        """
        if key not in self._data:
            raise KeyError(f"Metric '{key}' not initialized in MetricTracker.")

        self._data[key]["total"] += value * n
        self._data[key]["counts"] += n
        self._data[key]["average"] = self._data[key]["total"] / self._data[key]["counts"]

        # Optional logging
        if self.writer is not None:
            self.writer.add_scalar(key, value)

    def avg(self, key):
        """
        Return average value for a given metric.
        """
        return self._data[key]["average"]

    def result(self):
        """
        Return dict with average value of each metric.
        """
        return {key: vals["average"] for key, vals in self._data.items()}

    def keys(self):
        """
        Return all metric names.
        """
        return list(self._data.keys())
