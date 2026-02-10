from typing import List, Dict

from dataclasses import dataclass


@dataclass
class NormalisedData:

    data: Dict[str, List[List[float]]]

    @property
    def vehicles(self): return self.data['vehicles']

    @property
    def widths(self): return self.data['widths']

    @property
    def angles(self): return self.data['angles']

    @property
    def offsets(self): return self.data['offsets']

    @property
    def positions(self): return self.data['positions']

    @property
    def velocities(self): return self.data['velocities']

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.vehicles)

    def longest_track_length(self):
        return max([len(x) for x in self.widths])

    def normals_count(self):
        return sum([len(x) for x in self.widths])

    def vehicle_size(self):
        return len(self.vehicles[0])

    def track_length(self, t_idx):
        return len(self["widths"][t_idx])
