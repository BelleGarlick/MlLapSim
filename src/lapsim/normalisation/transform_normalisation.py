import json
import threading
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
from pydantic import BaseModel, Field

from lapsim.normalisation.normalisation_bounds import NormalisationBounds
from lapsim.encoder.partition import Partition
from lapsim.normalisation.transforms.transformer import Transform


"""This module contains the transform normalisation. This is the main object to
be used for normalising and transforming the input data to the network."""


class TransformNormalisation(BaseModel):

    transform: Transform = Field(default_factory=lambda: Transform())
    bounds: NormalisationBounds = Field(default_factory=lambda: NormalisationBounds())

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

    def extend(self, partition: Partition):
        """Extend the normalisation bounds based on the given partition"""
        vehicles = self.transform.vectorise_vehicles(partition.vehicles)
        self.bounds.extend(partition, vehicles)

        return self

    def normalise_and_transform(self, partition: Partition, cores: int = 1):
        """Normalise and transform the data"""
        vehicles = self.transform.vectorise_vehicles(partition.vehicles)
        normalisation = self.bounds.normalise(partition, vehicles)

        return self.transform.transform(normalisation, cores=cores)

    def detransform_and_denormalise(
            self,
            track_length: int,
            position: List[np.ndarray],
            velocity: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        position, velocity = self.transform.detransform(track_length, [position, velocity])
        return (
            position,
            velocity * (self.bounds.max_velocity - self.bounds.min_velocity) + self.bounds.min_velocity
        )

    def async_load_and_normalise_partition(self, partition_path: Union[str, Path], cores: int = 1):
        """Load and normalise a partition asyncronously

        Args:
            partition_path: File path to the partition

        Returns:
            The async partition loader object
        """
        loader = AsyncPartitionNormalisationLoader(partition_path, self, cores)
        loader.start()

        return loader


class AsyncPartitionNormalisationLoader(threading.Thread):
    """Helper object for loading and normalising the partition asyncronously"""

    def __init__(self, path: str, normaliser: TransformNormalisation, cores: int = 1):
        super().__init__()

        self._path = path
        self._normaliser = normaliser

        self.partition = None
        self.normalisation = None
        self.cores = cores

    def run(self):
        self.partition = Partition.load(self._path)
        self.normalisation = self._normaliser.normalise_and_transform(self.partition, cores=self.cores)
