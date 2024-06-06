from enum import Enum
from dataclasses import dataclass

class MoveType(Enum):

    RIGHT = 8 
    UP = 10 
    LEFT = 9 
    DOWN = 11 
    EXTRACT = 7 
    IMPLANT = 6 

@dataclass
class Move:

    move_type: MoveType
    index_x: int
    index_y: int
    block_size: int
    extraction_extent: int
