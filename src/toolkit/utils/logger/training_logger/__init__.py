import math
from typing import List, Optional, Callable

import numpy as np

from toolkit.utils.logger.training_logger.history import History

"""This module contains the logging object to print the training and validation
performance.

Example usage:
```
logger = Logger(labels=['Pos', 'Vel'], n_partitions=20)

for epoch in range(50):
    for i, partition in enumerate(partitions):
        for bidx, batch in enumerate(batches):
            p, v = model(batch)
            ploss, vloss = loss(p, _), loss(v, _)
            logger.write(50, bidx, len(batches), [ploss, vloss], partition=i)

    for bidx, batch in enumerate(vbatches):
        logger.write_val(epoch, bidx, len(vbatches))

    logger.flush()

print(logger.history.training['Pos'])
print(logger.history.training['Vel'])
print(logger.history.validation['Pos'])
print(logger.history.validation['Vel'])
```
"""


class Logger:
    """The logging object useful for understanding training/validation performance"""

    def __init__(self, labels: List[str] = None, n_partitions: int = 1, log_every: int = 10):
        self.n_partitions = n_partitions
        self.labels = labels or ['Loss']
        self.log_every = log_every

        self.history = History()

        self.avg_loss, self.avg_val_loss = {}, {}
        self._reset()

        self.best_val_loss = math.inf
        self.__on_new_best_val_loss_callback: Optional[Callable[[float], None]] = None

    def _reset(self):
        """Reset the current epochs data"""
        self.avg_loss = {label: [] for label in self.labels}
        self.avg_val_loss = {label: [] for label in self.labels}

    def write(self, epoch: int, batch: int, n_batches: int, losses: List[float], partition: int = -1):
        """Write training data"""
        for i, l in enumerate(losses):
            self.avg_loss[self.labels[i]].append(l)

        if (batch + 1) % self.log_every == 0:
            partition_items = (
                [f"Partition: {partition + 1}/{self.n_partitions}"]
                if self.n_partitions > 1 else
                []
            )

            print(", ".join(
                [f"\rTraining. Epoch: {epoch + 1}"]
                + partition_items
                + [f"Batch: {batch + 1}/{n_batches}"]
                + ["{}: {:.6f}".format(label, np.mean(self.avg_loss[label])) for label in self.labels]
                + [" " * 40]
            ))

    def write_val(self, epoch: int, batch: int, n_batches: int, losses: List[float]):
        """Write validation items"""
        for i, l in enumerate(losses):
            self.avg_val_loss[self.labels[i]].append(l)

        if (batch + 1) % self.log_every == 0:
            print(", ".join(
                [f"\rValidating. Epoch: {epoch + 1}"]
                + [f"Batch: {batch + 1}/{n_batches}"]
                + ["{}: {:.6f}".format(label, np.mean(self.avg_val_loss[label])) for label in self.labels]
                + [" " * 40]
            ))

    def flush(self, epoch: int):
        """Print the past epochs logs and add item to history"""
        print(", ".join(
            [f"\rEpoch: {epoch + 1}"]
            + ["{}: {:.6f}".format(label, np.mean(self.avg_loss[label])) for label in self.labels]
            + ["{}: {:.6f}".format(label, np.mean(self.avg_val_loss[label])) for label in self.labels]
        ))

        # Check if there's a new best validation loss
        if self.avg_val_loss:
            total_val_loss = sum([np.mean(x) for x in self.avg_val_loss.values()])
            if total_val_loss < self.best_val_loss:
                self.best_val_loss = total_val_loss
                print(f"New best val loss: {total_val_loss}")
                if self.__on_new_best_val_loss_callback:
                    self.__on_new_best_val_loss_callback(total_val_loss)

        # Write avg to history
        for label in self.labels:
            self.history.write('train', label, float(np.mean(self.avg_loss[label])))
            self.history.write('val', label, float(np.mean(self.avg_val_loss[label])))

        self._reset()

    def set_best_val_loss_callback(self, callback: Callable[[float], None]):
        """Set a callback function for when a new best val loss is reached"""
        self.__on_new_best_val_loss_callback = callback
