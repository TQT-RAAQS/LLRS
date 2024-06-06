from enum import Enum
from dataclasses import dataclass

class MoveType(Enum):

    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    EXTRACT = 4
    IMPLANT = 5

@dataclass
class Move:

    move_type: MoveType
    index_x: int
    index_y: int
    block_size: int
    extraction_extent: int
