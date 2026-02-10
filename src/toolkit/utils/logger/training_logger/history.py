from typing import Dict, List


"""The history class for the logger. This allows the logger to write to the
full history. This allows the user to get the history and get the full history
of the training and validation. 
"""


class History:
    def __init__(self):
        self.training: Dict[str, List[float]] = {}
        self.validation: Dict[str, List[float]] = {}

    def write(self, method: str, label: str, value: float):
        """Log an item to the history

        Args:
            method: Either 'train' or 'val'
            label: The label to tory in the history
            value: The value to log
        """
        assert method in {'train', 'val'}

        target_dict = self.training if method == 'train' else self.validation

        if label not in target_dict:
            target_dict[label] = []
        target_dict[label].append(value)
