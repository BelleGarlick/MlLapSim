from dataclasses import dataclass
from typing import Optional, Callable, List, Any

from toolkit.tracks.models.track import Track


"""The input parameters for splicing the optimal path into the track"""


@dataclass
class PathInput:

    x: float
    y: float
    vel: Optional[float]
    acc: Optional[float]


@dataclass
class SplicerInput:

    track: Track

    path: Optional[List[PathInput]] = None

    precision: Optional[int] = None

    on_complete_args: List[Any] = None
    on_complete: Callable[[Any], None] = None
