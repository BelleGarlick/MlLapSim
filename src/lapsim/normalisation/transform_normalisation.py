import json
import math
import os
import threading
from pathlib import Path
from typing import Union

import numpy as np
from pydantic import BaseModel, Field

from lapsim.normalisation.transforms.transformer import Transform


"""This module contains the transform normalisation. This is the main object to
be used for normalising and transforming the input data to the network."""


class TransformNormalisation(BaseModel):

    transform: Transform = Field(default_factory=lambda: Transform())

    # Normalisation bounds
    max_width: float = Field(default_factory=lambda: -math.inf)
    min_width: float = Field(default_factory=lambda: math.inf)
    max_angle: float = 0
    max_offset: float = 0
    max_velocity: float = Field(default_factory=lambda: -math.inf)
    min_velocity: float = Field(default_factory=lambda: math.inf)
    max_vehicle: list[float] | None = None
    min_vehicle: list[float] | None = None

    def extend(self, record: dict):
        vehicle = self.transform.transform_vehicle(record["vehicle"])

        self.max_angle = max(self.max_angle, np.max(np.abs(record["angles"])))
        self.max_offset = max(self.max_offset, np.max(np.abs(record["offsets"])))
        self.min_width = min(self.min_width, np.min(np.abs(record["widths"])))
        self.max_width = max(self.max_width, np.max(np.abs(record["widths"])))
        self.min_velocity = min(self.min_velocity, np.min(np.abs(record["vel"])))
        self.max_velocity = max(self.max_velocity, np.max(np.abs(record["vel"])))

        if not self.min_vehicle:
            self.min_vehicle = vehicle.tolist()
            self.max_vehicle = vehicle.tolist()
        else:
            for i in range(len(vehicle)):
                self.min_vehicle[i] = min(self.min_vehicle[i], vehicle[i])
                self.max_vehicle[i] = max(self.max_vehicle[i], vehicle[i])

    def save(self, path):
        with open(path, "w+") as file:
            file.write(self.model_dump_json(exclude_none=True))

    @classmethod
    def load(cls, path):
        with open(path) as file:
            data = json.load(file)

            # Loop through various keys and if they're in the data
            # but None then remove the key
            if 'bounds' in data:
                for key in {'max_width', 'min_width', 'min_velocity', 'max_velocity'}:
                    if key in data['bounds'] and data['bounds'][key] is None:
                        del data['bounds'][key]

            return TransformNormalisation.model_validate(data)

    def detransform_and_denormalise(
            self,
            track_length: int,
            position: list[np.ndarray],
            velocity: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        position, velocity = self.transform.detransform(track_length, [position, velocity])
        return (
            position,
            velocity * (self.max_velocity - self.min_velocity) + self.min_velocity
        )

    def async_load_and_normalise_partition(self, partition_path: Union[str, Path], cache_tensor_prefix=None, cores: int = 1):
        """Load and normalise a partition asyncronously

        Args:
            partition_path: File path to the partition

        Returns:
            The async partition loader object
        """
        loader = AsyncPartitionNormalisationLoader(partition_path, self, cache_tensor_prefix, cores)
        loader.start()

        return loader

    def normalise(self, record):
        return {
            'widths': (record["widths"] - self.min_width) / (self.max_width - self.min_width),
            'angles': record["angles"] / self.max_angle,
            'offsets': record["offsets"] / self.max_offset,

            'vehicle': (self.transform.transform_vehicle(record["vehicle"]) - self.min_vehicle) / (
                        np.array(self.max_vehicle) - np.array(self.min_vehicle)),

            # 'positions': record["positions"],
            # 'velocities': (record["velocities"] - self.min_velocity) / (self.max_velocity - self.min_velocity),

            # TODO deprecate
            'pos': record["pos"],
            'vel': (record["vel"] - self.min_velocity) / (self.max_velocity - self.min_velocity),
        }


class AsyncPartitionNormalisationLoader(threading.Thread):
    """Helper object for loading and normalising the partition asyncronously"""

    def __init__(self, path: str, normaliser: TransformNormalisation, cache_tensor_prefix=None, cores: int = 1):
        super().__init__()

        self._path = path
        self._normaliser = normaliser
        self.cache_tensor_prefix = cache_tensor_prefix

        self.normalisation = None
        self.cores = cores

    def run(self):
        if self.cache_tensor_prefix is not None and os.path.exists(self.cache_tensor_prefix + "-x.npy"):
            x = np.load(self.cache_tensor_prefix + f"-x.npy")
            y_pos = np.load(self.cache_tensor_prefix + f"-ypos.npy")
            y_vel = np.load(self.cache_tensor_prefix + f"-yvel.npy")
            vehicles = np.load(self.cache_tensor_prefix + f"-v.npy")
        else:
            partition = Partition.load(self._path)
            x, (y_pos, y_vel), vehicles = self._normaliser.normalise_and_transform(partition, cores=self.cores)

            if self.cache_tensor_prefix:
                np.save(self.cache_tensor_prefix + f"-x.npy", x)
                np.save(self.cache_tensor_prefix + f"-ypos.npy", y_pos)
                np.save(self.cache_tensor_prefix + f"-yvel.npy", y_vel)
                np.save(self.cache_tensor_prefix + f"-v.npy", vehicles)

        self.normalisation = x, (y_pos, y_vel), vehicles
