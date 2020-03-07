from enum import Enum
from typing import TypedDict


class Category(Enum):
    SecondHand = 0
    ResidentialLand = 1
    ResidentialLandWithConstruction = 2
    WoodLand = 3
    FarmLand = 4


Floor = TypedDict(
    "Floor",
    {"room": int, "dinning": int, "living": int, "kitchen": int, "storage": int,},
)


Quater = TypedDict("Quater", {"year": int, "quarter": int,})
