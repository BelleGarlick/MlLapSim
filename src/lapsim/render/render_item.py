from dataclasses import dataclass
from typing import Optional

from toolkit.tracks.models import Track


@dataclass
class RenderItem:
    track: dict
    label: str
    color: Optional[str]
