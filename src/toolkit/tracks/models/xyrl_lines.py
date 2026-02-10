from pydantic import Field, BaseModel


class XYRLLine(BaseModel):

    x: float = Field(description="X ordinate of the reference line")
    y: float = Field(description="Y ordinate of the reference line")
    r: float = Field(description="Length of the right side of the normal line")
    l: float = Field(description="Length of the left side of the normal line")
