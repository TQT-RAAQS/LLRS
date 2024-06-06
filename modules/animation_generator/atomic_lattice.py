import numpy as np
from typing import List, Tuple

from .trap import Trap, AtomState
from .move import Move, MoveType

class AtomicLattice:

    traps: List[List[Trap]]
    N_x: int
    N_y: int

    def __init__(self, N_x: int, N_y: int):
        self.N_x = N_x
        self.N_y = N_y

        self.traps = np.full((N_y, N_x) ,np.nan, dtype = Trap)
        for i in range(N_x):
            for j in range(N_y):
                self.traps[j, i] = Trap()

    def populate_traps_by_matrix(self, flags: List[List[bool]], atom_state: AtomState = AtomState.STATIC_TRAP):
        assert flags.shape == self.traps.shape
        for i in range(len(self.traps.shape[0])):
            for j in range(len(self.traps.shape[1])):
                self.set_trap(j, i, atom_state)

    def populate_traps_by_list(self, indices: List[Tuple], atom_state: AtomState = AtomState.STATIC_TRAP):
        for x, y in indices:
            self.get_trap(x, y).occupy_with_atom(atom_state)

    def empty_traps(self):
        for i in range(self.N_y):
            for j in range(self.N_x):
                self.get_trap(j, i).empty_trap()

    def set_trap(self, xind: int, yind: int, atom_state: AtomState):
        self.traps[yind, xind].set_atomic_state(atom_state)

    def get_trap(self, xind: int, yind: int) -> Trap:
        return self.traps[yind, xind]
    
    def apply_move(self, move: Move):
        self._verify_move(move)
        
        if move.move_type ==  MoveType.RIGHT:
            return self._right(move)
        elif move.move_type ==  MoveType.UP:
            return self._up(move)
        elif move.move_type ==  MoveType.LEFT:
            return self._left(move)
        elif move.move_type ==  MoveType.DOWN:
            return self._down(move)
        elif move.move_type ==  MoveType.EXTRACT:
            return self._extract(move)
        elif move.move_type ==  MoveType.IMPLANT:
            return self._implant(move)
        else:
            raise NotImplementedError(f"The move {move} is not handled.")
        
    def print(self):
        for i in range(self.N_y):
            for j in range(self.N_x):
                print(self.get_trap(j, i).get_atom_state().value if self.get_trap(j, i).is_occupied() else 0, end=" ")
            print()

    def _extract(self, move: Move):
        for i in range(move.block_size):
            self.get_trap(move.index_x, move.index_y + i).occupy_with_atom(AtomState.DYNAMIC_TRAP)

        return [(move.index_x, move.index_y + i) for i in range(move.block_size)]

    def _implant(self, move: Move):
        for i in range(move.block_size):
            self.get_trap(move.index_x , move.index_y + i).occupy_with_atom(AtomState.STATIC_TRAP)

        return [(move.index_x, move.index_y + i) for i in range(move.block_size)]

    def _right(self, move: Move):
        for i in range(move.block_size):
            self.get_trap(move.index_x, move.index_y + i).empty_trap()
        for i in range(move.block_size):
            self.get_trap(move.index_x + 1, move.index_y + i).occupy_with_atom(AtomState.DYNAMIC_TRAP)

        return [(move.index_x, move.index_y + i) for i in range(move.block_size)]

    def _left(self, move: Move):
        for i in range(move.block_size):
            self.get_trap(move.index_x, move.index_y + i).empty_trap()
        for i in range(move.block_size):
            self.get_trap(move.index_x - 1, move.index_y + i).occupy_with_atom(AtomState.DYNAMIC_TRAP)

        return [(move.index_x, move.index_y + i) for i in range(move.block_size)]

    def _up(self, move: Move):
        self.get_trap(move.index_x, move.index_y - 1).occupy_with_atom(AtomState.DYNAMIC_TRAP)
        self.get_trap(move.index_x, move.index_y + move.block_size - 1).empty_trap()

        return [(move.index_x, move.index_y + i) for i in range(move.block_size)]

    def _down(self, move: Move):
        self.get_trap(move.index_x, move.index_y).empty_trap()
        self.get_trap(move.index_x, move.index_y + move.block_size).occupy_with_atom(AtomState.DYNAMIC_TRAP)

        return [(move.index_x, move.index_y + i) for i in range(move.block_size)]

    def _verify_move(self, move: Move):
        assert move.index_y + move.block_size - 1 < self.traps.shape[1], "The designated block of traps does not exist"

        for i in range(move.block_size):
            x, y = move.index_x, move.index_y + i

            assert self.get_trap(x, y).is_occupied(),\
                f"The trap has no atoms, and thus impossible to act upon with this move: {move}"
            
            if move.move_type in [MoveType.EXTRACT]:
                assert self.get_trap(x, y).get_atom_state() == AtomState.STATIC_TRAP,\
                    f"All atoms must be already in the static traps for this move to be valid: {move}"
            else:
                assert self.get_trap(x, y).get_atom_state() == AtomState.DYNAMIC_TRAP,\
                    f"All atoms must be already in the dynamic traps for this move to be valid: {move}"
                
            if move.move_type ==  MoveType.RIGHT:
                assert x < self.N_x - 1, "Cannot move the last column to the right"
                assert not self.get_trap(x + 1, y).is_occupied(), "Cannot move an atom to an occupied trap"
            elif move.move_type ==  MoveType.LEFT:
                assert x > 0, "Cannot move the first column to the left"
                assert not self.get_trap(x - 1, y).is_occupied(), "Cannot move an atom to an occupied trap"
        
        x, y = move.index_x, move.index_y
        if move.move_type ==  MoveType.DOWN:
            assert y + move.block_size < self.N_y, "Cannot move the last row down"
            assert not self.get_trap(x, y + move.block_size).is_occupied(), "Cannot move an atom to an occupied trap"
        elif move.move_type ==  MoveType.UP:
            assert y > 0, "Cannot move the first column up"
            assert not self.get_trap(x, y - 1).is_occupied(), "Cannot move an atom to an occupied trap"
                
        assert move.extraction_extent <= self.traps.shape[0] and move.extraction_extent >= 0,\
            f"Invalid extraction extent: {move.extraction_extent}"