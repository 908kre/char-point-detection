from enum import Enum
import typing as t


class Category(Enum):
    SecondHand = 0
    ResidentialLand = 1
    ResidentialLandWithConstruction = 2
    WoodLand = 3
    FarmLand = 4


Floor = t.TypedDict(
    "Floor",
    {
        "room": t.Optional[int],
        "dinning": t.Optional[int],
        "living": t.Optional[int],
        "kitchen": t.Optional[int],
        "storage": t.Optional[int],
    },
)


Quater = t.TypedDict("Quater", {"year": int, "quarter": int,})
