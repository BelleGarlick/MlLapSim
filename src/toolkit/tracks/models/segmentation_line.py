import math
from typing import List, Optional

from pydantic import BaseModel, Field


class SegmentationLine(BaseModel):

    class Config:
        json_schema_extra = {"description": "A track normal line."}

    x1: float = Field(description="Left X ordinate of the normal line")
    y1: float = Field(description="Left Y ordinate of the normal line")
    x2: float = Field(description="Right X ordinate of the normal line")
    y2: float = Field(description="Right Y ordinate of the normal line")

    pos: Optional[float] = Field(
        default=None,
        description="The relative position of the vehicle upon this line.")
    vel: Optional[float] = Field(
        default=None,
        description="The velocity of the vehicle as it passes this line.")
    acc: Optional[float] = Field(
        default=None,
        description="The acceleration of the vehicle as it passes this line.")

    def arr(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def length(self):
        return math.hypot(self.y1 - self.y2, self.x1 - self.x2)
