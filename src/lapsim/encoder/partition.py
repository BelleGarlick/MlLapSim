import threading
from pathlib import Path
from typing import List, Union

from pydantic import BaseModel


class Partition(BaseModel):

    vehicles: List[dict]

    widths: List[List[float]]
    angles: List[List[float]]
    offsets: List[List[float]]

    positions: List[List[float]]
    velocities: List[List[float]]

    @staticmethod
    def load(path: str) -> 'Partition':
        with open(path) as file:
            return Partition.model_validate_json(file.read())

    @staticmethod
    def async_load(path: str):
        partition = AsyncPartitionLoader(path)
        partition.start()

        return partition

    def save(self, path: Union[Path, str]):
        print(f"Saving to {path}")
        with open(path, "w+") as file:
            file.write(self.model_dump_json())

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
