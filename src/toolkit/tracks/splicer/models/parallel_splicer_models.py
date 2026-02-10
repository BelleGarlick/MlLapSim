from typing import List

from pydantic import BaseModel

from toolkit.tracks.splicer.models.splicer_input import SplicerInput
from toolkit.tracks.models import Track


class ParallelSplicerInput(BaseModel):
    splicer_input: SplicerInput
    return_output: bool


class ParallelSplicerOutput(BaseModel):
    spliced: List[Track]
    errors: List[str]
