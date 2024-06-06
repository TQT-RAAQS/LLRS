from enum import Enum

class AtomState(Enum):
    
    STATIC_TRAP = 1 # The atom is in the static trap (not extracted)
    DYNAMIC_TRAP = 2 # The atom is in the dynamic traps (extracted)

class Trap:

    occupied: bool = False
    atom_state: AtomState = None

    def occupy_with_atom(self, atom_state: AtomState):
        self.occupied = True
        self.atom_state = atom_state

    def empty_trap(self):
        self.occupied = False
        self.atom_state = None

    def is_occupied(self):
        return self.occupied
    
    def get_atom_state(self):
        return self.atom_state

