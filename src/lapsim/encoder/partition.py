import threading
from pathlib import Path
from typing import List, Union

from pydantic import BaseModel, Field


class Partition(BaseModel):

    vehicles: List[dict] = Field(default_factory=list)

    widths: List[List[float]] = Field(default_factory=list)
    angles: List[List[float]] = Field(default_factory=list)
    offsets: List[List[float]] = Field(default_factory=list)

    positions: List[List[float]] = Field(default_factory=list)
    velocities: List[List[float]] = Field(default_factory=list)

    @staticmethod
    def load(path: Union[str, Path]) -> 'Partition':
        with open(path) as file:
            return Partition.model_validate_json(file.read())

    @staticmethod
    def async_load(path: str):
        partition = AsyncPartitionLoader(path)
        partition.start()

        return partition

    def save(self, path: Union[Path, str]):
        with open(path, "w+") as file:
            file.write(self.model_dump_json())

    def append(self, partitions: Partition):
        self.vehicles.extend(partitions.vehicles)
        self.widths.extend(partitions.widths)
        self.angles.extend(partitions.angles)
        self.offsets.extend(partitions.offsets)
        self.positions.extend(partitions.positions)
        self.velocities.extend(partitions.velocities)

    @staticmethod
    def combine(partitions: List['Partition']):
        vehicles = []
        widths, angles, offsets = [], [], []
        positions, velocities = [], []

        for partition in partitions:
            vehicles += partition.vehicles

            widths += partition.widths
            angles += partition.angles
            offsets += partition.offsets

            positions += partition.positions
            velocities += partition.velocities

        return Partition(
            vehicles=vehicles,

            widths=widths,
            angles=angles,
            offsets=offsets,

            positions=positions,
            velocities=velocities
        )


class AsyncPartitionLoader(threading.Thread):
    def __init__(self, path: str):
        super().__init__()

        self.path = path
        self.partition = None

    def run(self):
        self.partition = Partition.load(self.path)
