from typing import List

from pydantic import BaseModel, Field

from toolkit.tracks.models.segmentation_line import SegmentationLine


class Track(BaseModel):

    segmentations: List[SegmentationLine] = Field(
        title="Track normal lines",
        description="The list of lines which define the track geometry. These "
                    + "initially generated normal to the track center line "
                    + "then smoothed out to prevent intersections."
    )

    class Config:
        title = 'The Track object'
        json_schema_extra = {
            "description": "The track object, containing a list of lines defining the track geometry."
        }

    def midline(self) -> List[List[float]]:
        return [
            [
                (normal.x1 + normal.x2) / 2,
                (normal.y1 + normal.y2) / 2
            ]
            for normal in self.segmentations
        ]

    def left_line(self) -> List[List[float]]:
        return [
            [normal.x1, normal.y1]
            for normal in self.segmentations
        ]

    def right_line(self) -> List[List[float]]:
        return [
            [normal.x2, normal.y2]
            for normal in self.segmentations
        ]

    @staticmethod
    def from_file(path):
        with open(path) as file:
            return Track.model_validate_json(file.read())
