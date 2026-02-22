from typing import List

from pydantic import BaseModel

from toolkit.tracks.splicer.models.splicer_input import SplicerInput
from toolkit.tracks.models import Track

class ParallelSplicerOutput(BaseModel):
    spliced: List[Track]
    errors: List[str]
