import abc
import math
from typing import List

import numpy as np
from toolkit import maths


# TODO Requires documenting and testing


class SpatialMapItem(abc.ABC):
    def center(self):
        raise NotImplementedError


class SpatialLineItem(SpatialMapItem):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

        self.min_x = min(p1[0], p2[0])
        self.max_x = max(p1[0], p2[0])
        self.min_y = min(p1[1], p2[1])
        self.max_y = max(p1[1], p2[1])

        self._center = (
            (p1[0] + p2[0]) / 2,
            (p1[1] + p2[1]) / 2
        )

    def center(self):
        return self._center

    def to_line(self):
        return [self.p1[0], self.p1[1], self.p2[0], self.p2[1]]

    def collisision(self, line):
        # TODO Need to check the min/max x/y's
        intersections = maths.segment_intersections(line.to_line(), [self.to_line()])
        if intersections:
            return intersections[0]

        return None


class SpatialMap:
    def __init__(self, cell_size: float):
        self.items: List[SpatialLineItem] = []
        self.map = {}
        self.cell_size = cell_size

        self.cached_boxes = []
        self.cached_cell = None

    def _get_cell_pos_from_position(self, pos):
        return (
            math.floor(pos[0] / self.cell_size),
            math.floor(pos[1] / self.cell_size)
        )

    def add_item(self, item: SpatialMapItem):
        cell_pos = self._get_cell_pos_from_position(item.center())
        if cell_pos[0] not in self.map:
            self.map[cell_pos[0]] = {}

        if cell_pos[1] not in self.map[cell_pos[0]]:
            self.map[cell_pos[0]][cell_pos[1]] = []

        box_idx = len(self.items)
        self.items.append(item)
        self.map[cell_pos[0]][cell_pos[1]].append(box_idx)

        return self

    def get_items(self, position: np.ndarray) -> List[int]:  # TODO, does this need to be numpy
        box_indexes = []

        cell_pos = self._get_cell_pos_from_position(position)

        # v minor optimisation to cache the cell and return the
        # last boxes, this works because the boxes are iteratively
        # looped through. And saves a few lookups and array rebuilds
        if np.all(cell_pos == self.cached_cell):
            return self.cached_boxes

        x, y = int(cell_pos[0]), int(cell_pos[1])
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                box_indexes += self.map.get(i, {}).get(j, [])

        self.cached_cell = cell_pos
        self.cached_boxes = box_indexes

        return box_indexes
