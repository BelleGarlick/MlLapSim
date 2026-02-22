import math
from typing import Optional, List

import numpy as np
from pydantic import Field, BaseModel

from lapsim.normalisation.normalised_data import NormalisedData
from lapsim.encoder.partition import Partition


"""The normalisation bounds object.

This object shouldn't be called directly but should instead be called via the 
`TransformNormalisation`. This object contains functionality for taking the 
encoded track and normalising it between 0 - 1 and -1 - 1."""


def range_normalise(values: List[List[float]], _max: float, _min: float) -> List[List[float]]:
    """Shift and divide a list of lists of float from a range to 0 - 1"""
    if _max == -math.inf or _min == math.inf:
        raise Exception("Normalisation bounds not loaded")

    divisor = _max - _min
    if divisor == 0:
        return [[0 for _ in x] for x in values]

    return [
        [(v - _min) / divisor for v in x]
        for x in values
    ]


def scalar_normalise(values: List[List[float]], _scale: float) -> List[List[float]]:
    """Divide a list of lists by a specified number"""
    if _scale == 0:
        return [[0 for _ in x] for x in values]

    return [
        [v / _scale for v in x]
        for x in values
    ]


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

    def normalise(self, partition: Partition, vehicles: List[List[float]]) -> NormalisedData:
        """Normalise a partition and vehicle data.

        This function takes all the values from the partition and normalises
        them for each data type.

        This function should only be called from the TransformNormalisation
        object which vectorises the vehicles and passes it to this function.

        Args:
            partition: The partition to encode.
            vehicles: A list of vectorised vehicles.
        """
        return NormalisedData({
            "widths": range_normalise(partition.widths, _max=self.max_width, _min=self.min_width),
            "angles": scalar_normalise(partition.angles, _scale=self.max_angle),
            "offsets": scalar_normalise(partition.offsets, _scale=self.max_offset),

            "positions": partition.positions,
            "velocities": range_normalise(partition.velocities, _max=self.max_velocity, _min=self.min_velocity),

            "vehicles": self._normalise_vehicles(vehicles)
        })

    def _normalise_vehicles(self, vehicles: List[List[float]]) -> List[List[float]]:
        """Normalise the list of vectorised vehicles and converts it to 0 - 1."""
        if self.max_vehicle is None or self.min_vehicle is None:
            raise Exception("Normalisation bounds not loaded.")

        # Calculate the difference between each item in the vehicle bounds
        divisor = [y - x for x, y in zip(self.min_vehicle, self.max_vehicle)]

        # If any value in the divisor is 0, set it to 1 to avoid a div0 error
        divisor = [1 if x == 0 else x for x in divisor]

        norm_bounds = list(zip(self.min_vehicle, divisor))

        return [
            [
                (float(v) - float(v_min)) / div
                for v, (v_min, div) in list(zip(vehicle, norm_bounds))
            ]
            for vehicle in vehicles
        ]
