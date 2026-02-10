from collections import defaultdict
from typing import Dict, List

import numpy as np

"""
Logging function for training. See training notebooks for usage.
"""


class Logger:
    def __init__(self, n_partitions, callbacks=None):
        self.current_train_percentage = 0

        self.epoch_training_metrics: Dict[str, List[float]] = defaultdict(lambda: [])
        self.history_training_metrics: Dict[str, List[float]] = defaultdict(lambda: [])

        self.avg_val_pos_loss = []
        self.avg_val_vel_loss = []

        self.n_partitions = n_partitions

        self.callbacks = callbacks or []

        self._last_percentage = None

    def log_training_metrics(self, percentage: float, **kwargs: Dict[str, float]):
        self.current_train_percentage = percentage

        for key in kwargs:
            self.epoch_training_metrics[key].append(kwargs[key])

        # Call callbacks
        for callback in self.callbacks:
            callback.on_training_loss_registered(self)

        # Create training log
        percentage = round(percentage, 3)
        if percentage != self._last_percentage:
            self._last_percentage = percentage
            message = f"Training {int(percentage * 100)}%"
            for key in self.epoch_training_metrics:
                message += f" {key}: {np.mean(self.epoch_training_metrics[key]):.5f}"
            print(f"\r{message}", end="")

    def log_val(self, epoch, batch, batches_n, position_loss, velocity_loss):
        self.avg_val_pos_loss += [position_loss]
        self.avg_val_vel_loss += [velocity_loss]

        if batch % 10 == 0:
            print("\rValidating. Epoch {}, Batch: {}/{}. Pos Loss: {:.6f} Vel Loss: {:.6f}{}".format(
                epoch + 1,
                batch + 1, batches_n,
                np.mean(self.avg_val_pos_loss),
                np.mean(self.avg_val_vel_loss),
                " " * 20
            ), end="")

    def flush(self, epoch):
        self.history_training_metrics['pos_loss'].append(np.mean(self.epoch_training_metrics['pos_loss']))
        self.history_training_metrics['vel_loss'].append(np.mean(self.epoch_training_metrics['vel_loss']))

        print("\rEpoch: {}: Pos Loss: {:.6f}, Vel Loss: {:.6f}, Val Pos Loss: {:.6f}, Val Vel Loss: {:.6f}".format(
            epoch + 1,
            self.history_training_metrics['pos_loss'][-1],
            self.history_training_metrics['vel_loss'][-1],
            np.mean(self.avg_val_pos_loss),
            np.mean(self.avg_val_vel_loss),
        ))
        self.avg_val_pos_loss = []
        self.avg_val_vel_loss = []
        self.epoch_training_metrics.clear()
