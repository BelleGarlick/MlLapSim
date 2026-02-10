from typing import List

from pydantic import BaseModel

from toolkit.tracks.models.xyrl_lines import XYRLLine


class XYRLTrack(BaseModel):

    normals: List[XYRLLine]
