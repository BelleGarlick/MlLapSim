from dataclasses import dataclass

from toolkit.tracks.models import Track


@dataclass
class EncoderInput:

    track: Track
    vehicle: dict

    flip: bool = False
