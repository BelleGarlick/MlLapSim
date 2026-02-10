from typing import Union, Optional, List

import numpy as np
from pydantic import BaseModel, Field

from lapsim.normalisation.normalised_data import NormalisedData

from lapsim.normalisation.transforms.bidirectional import BidirectionalTransformMethod
from lapsim.normalisation.transforms.lagging import LaggingTransformMethod, StatefulLaggingTransformMethod
from lapsim.normalisation.transforms.window import WindowTransform, FlatWindowTransform


VEHICLE_KEYS = {
    'V1': [
        "track_front", "track_rear", "wheel_base_front", "wheel_base_rear",
        "mass", "k_drive_front", "k_roll", "tyre_friction",
        "max_power", "cog_height", "lift_coeff_front", "lift_coeff_rear",
        "v_max", "drag_coeff", "yaw_inertia", "k_brake_front"
    ]
}


DEFAULT_INPUTS = ["widths", "angles", "offsets"]
DEFAULT_OUTPUTS = ["positions", "velocities"]


transform_map = {
    "bidirectional": BidirectionalTransformMethod(),
    "lag": LaggingTransformMethod(),
    "stateful-lag": StatefulLaggingTransformMethod(),
    "window": WindowTransform(),
    "flat-window": FlatWindowTransform()
}


class Transform(BaseModel):

    method: str = Field(default="window")
    vehicle_encoding: Union[str, List[str]] = Field(default="V1")

    inputs: List[str] = Field(default=DEFAULT_INPUTS)
    outputs: List[str] = Field(default=DEFAULT_OUTPUTS)

    # Required for classic window method
    foresight: Optional[int] = None
    sampling: Optional[int] = None

    lag: Optional[int] = None

    patch_size: int = 1
    time_to_vec: bool = False  # Should always be relative to the track rather than current track window otherwise will always learn to predict for when values equal 1
    random_repeats: int = 1  # Choose a random number up to this point
    decimation: float = 0  # 0.5 means half the randomly track disapears, this is to help overfitting

    def transform_vehicle(self, vehicle: dict) -> List[float]:
        """Transform a vehicle based on a specific order.

        This function is used to create a modular way of encoding vehicle
        information accross different models with different requirements. The
        encoding can either be directly states as a list of keys given in the
        `vehicle_encoding` argument or as a key from the predefined list
        `VEHICLE_KEYS` which should contain the vehicle encodings for the
        different published models.

        This method helps us keep backwards compatability as we change the
        models since we keep the ability to generate encodings for older models
        with different encoding requirements.

        Args:
            vehicle: The vehicle dictionary containing the keys to encode.

        Returns:
            The vehicle vector
        """
        keys_order = (
            VEHICLE_KEYS[self.vehicle_encoding]
            if isinstance(self.vehicle_encoding, str) else
            self.vehicle_encoding
        )

        lower_keyed_vehicle = {key.lower(): vehicle[key] for key in vehicle}

        # Check the vehicle for the different keys
        vehicle_data = []
        for snake_case_key in keys_order:
            camel_case_key = snake_case_key.replace("_", "").lower()

            if snake_case_key in lower_keyed_vehicle: vehicle_data.append(lower_keyed_vehicle[snake_case_key])
            elif camel_case_key in lower_keyed_vehicle: vehicle_data.append(lower_keyed_vehicle[camel_case_key])
            else:
                raise ValueError(f"Vehicle is missing attribute: '{snake_case_key}' or '{camel_case_key}'")

        return vehicle_data

    def vectorise_vehicles(self, vehicles: List[dict]) -> List[List[float]]:
        """Vectorise a list of vehicles"""
        return [self.transform_vehicle(x) for x in vehicles]

    def get_transform(self):
        """Get the transform method class, then set the params of the transform"""
        if self.method not in transform_map:
            raise Exception(f"Unknown transform method: '{self.method}'")

        transform = transform_map[self.method]

        transform.inputs = self.inputs
        transform.outputs = self.outputs

        transform.sampling = self.sampling
        transform.patch_size = self.patch_size
        transform.lag = self.lag
        transform.foresight = self.foresight
        transform.time_to_vec = self.time_to_vec

        return transform

    def transform(self, normalised_data: NormalisedData, cores: int):
        return self.get_transform().transform(normalised_data, cores)

    def detransform(self, track_length: int, outputs: List[np.ndarray]):
        return self.get_transform().detransform(track_length, outputs)
