import json
import math
from typing import Optional, List

import numpy as np
from pydantic import Field, BaseModel
from webdataset.compat import WebDataset

from lapsim.normalisation.normalised_data import NormalisedData
from lapsim.encoder.partition import Partition


"""The normalisation bounds object.

This object shouldn't be called directly but should instead be called via the 
`TransformNormalisation`. This object contains functionality for taking the 
encoded track and normalising it between 0 - 1 and -1 - 1."""


def get_max_from_lists(curr_value: float, items: List[List[float]]):
    """Get the maximum from an array, used for offsets and angles"""
    val = curr_value
    for item in items:
        if item:
            val = max(val, np.max(np.abs(item)))
    return val


def _get_min_and_max_from_lists(curr_min_value: float, curr_max_value: float, items: List[List[float]]):
    """Get the minimum and maximum from a list of lists.

    This function is used to extend the normalisation bounds.
    """
    min_val = curr_min_value
    max_val = curr_max_value
    for item in items:
        if item:
            min_val = min(min_val, np.min(np.abs(item)))
            max_val = max(max_val, np.max(np.abs(item)))
    return min_val, max_val


class NormalisationBounds(BaseModel):

    max_width: float = Field(default_factory=lambda: -math.inf)
    min_width: float = Field(default_factory=lambda: math.inf)
    max_angle: float = 0
    max_offset: float = 0
    max_velocity: float = Field(default_factory=lambda: -math.inf)
    min_velocity: float = Field(default_factory=lambda: math.inf)

    max_vehicle: Optional[List[float]] = None
    min_vehicle: Optional[List[float]] = None

    def extend(self, partition: Partition, vehicles: List[List[float]]):
        """Extend the normalisation bounds based on a partition

        Args:
            partition: The partition to get values from to normalise
            vehicles: The vectorised vehicles. These aren't taken from the
                partition as those are not vectorised yet.
        """
        # todo do this but better
        self.max_angle = get_max_from_lists(self.max_angle, partition.angles)
        self.max_offset = get_max_from_lists(self.max_offset, partition.offsets)

        self.min_width, self.max_width = _get_min_and_max_from_lists(
            self.min_width, self.max_width, partition.widths)

        self.min_velocity, self.max_velocity = _get_min_and_max_from_lists(
            self.min_velocity, self.max_velocity, partition.velocities)

        for vehicle in vehicles:
            if vehicle:
                if not self.min_vehicle:
                    self.min_vehicle = vehicle
                    self.max_vehicle = vehicle
                else:
                    v_arr = np.array([self.min_vehicle, self.max_vehicle, vehicle], np.float32)

                    self.min_vehicle, self.max_vehicle = (
                        np.min(v_arr, axis=0).tolist(), np.max(v_arr, axis=0).tolist())

