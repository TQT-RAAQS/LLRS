import sys
from typing import Dict, List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

from copy import deepcopy

from structures.graph import DynamicArrayGraph
from models.array_model import TrapArray


class Operation(object):
    '''
    Represents an operation that happens in one time step. This can imply multiple atoms
    moving simultaneously, or a single atom's move.

    Attributes:
    ----
        trap_movements : dict
            dict of from_trap - > to_trap

    '''

    def __init__(self, trap_movements: List[Tuple[int, int]]):
        '''
        Instantiates an Operation object

        Parameters
        ----------
        trap_movements: a dictionary {from_trap_idx: to_trap_idx, from_trap_idx: to_trap_idx, ...}

        '''
        
        self.trap_movements = {}

        for move in trap_movements:
            #print(move)
            self.trap_movements[move[0]] = move[1]
    
    def apply(self, static_array: TrapArray):
        '''
        Apply operation to static array.

        Args:
            static_array (TrapArray) : static array to apply operation to 
        
        '''
        
        from_trap_atoms = {}
        for trap_idx in self.trap_movements.keys():
            from_trap_atoms[self.trap_movements[trap_idx]] = static_array.get_trap_by_index(trap_idx).clear_trap()

        for from_trap_idx in from_trap_atoms:
            atoms = from_trap_atoms[from_trap_idx]
            static_array.get_trap_by_index(from_trap_idx).trap_atom(atoms)

    def __str__(self):
        string = 'Trap -> Destination Trap:'
        for trap in self.trap_movements.keys():
            string += '\n   ' + str(trap) + ' -> ' + str(self.trap_movements[trap])
        return string


class SubSequence(object):
    '''
    Represents a list of Operations performed in a ReconfigurationSequence between

    Attributes:
    ----
        sequence : List[Operation]
            list of operations performed
        
        graph : DynamicArrayGraph
            graph upon which abstract operations are performed

        initial_static_array_occupation_state : list
            list of occupied indices in initial state
        
        target_static_array_occupation_state : list
            list of occupied indices in target state

        final_static_array_occupation_state : list
            list of occupied indices in final state after measurement

        virtual_runtime : float
            time taken to compute operation list

        operation_conversion_time : float
            time taken to convert dict of operations to abstract operation object list

        aod_encoding_translation_time : float
            time taken to convert abstract operations to AOD encodings on a grid

    '''

    def __init__(self, graph: DynamicArrayGraph, initial_static_array_occupation_state, target_static_array_occupation_state):
        self.sequence = []
        self.graph = graph
        self.virtual_runtime = 0
        self.operation_conversion_time = 0
        self.aod_encoding_translation_time = 0
        self.initial_static_array_occupation_state=initial_static_array_occupation_state
        self.target_static_array_occupation_state=target_static_array_occupation_state
        self.final_static_array_occupation_state=None
        self.atoms_lost_after_measurement=[]

    def add_operation(self, op: Operation) -> None:
        '''
        Appends an Operation object to sequence. 

        Args:
            op (Operation) : operation to append

        Raises: 
            Exception: If operation between two traps not in dynamic array graph

        '''

        if self._verify_operation(op) and (len(op.trap_movements.keys()) > 0):
            #print('op added to seq:', op)
            self.sequence.append(op)
        else:
            Exception('Invalid operation provided.')

    def add_operations(self, ops: List[Operation]) -> None:
        for op in ops:
            #print('op from adding to reconfig seq: ', op)
            self.add_operation(op)

    def _verify_operation(self, op):
        '''
        Verify operation is valid for this instance of the problem.

        Args:
            op (Operation) : operation to append

        Returns:
            bool : whether operation is valid

        Raises: 
            Exception: If operation between two traps not in dynamic array graph

        '''

        for cur_trap in op.trap_movements.keys():
            if op.trap_movements[cur_trap] not in self.graph.get_edges(cur_trap):
                Exception('Atom traversed edge not in array.')
        
        return True

    def get_sequence(self) -> List[Operation]:
        if self.loss:
            return self.sub_sequences
        else:
            return self.sequence

    def set_atoms_lost_indices(self, atom_indices):
        self.atoms_lost_after_measurement=atom_indices

    def get_atoms_lost_indices(self):
        return self.atoms_lost_after_measurement

    def set_virtual_runtime(self, runtime):
        self.virtual_runtime = runtime

    def set_operation_conversion_time(self, runtime):
        self.operation_conversion_time = runtime

    def set_aod_encoding_translation_time(self, runtime):
        self.aod_encoding_translation_time = runtime

    def get_initial_state(self):
        return self.initial_static_array_occupation_state
    
    def get_target_state(self):
        return self.target_static_array_occupation_state

    def get_virtual_runtime(self):
        return self.virtual_runtime

    def get_operation_conversion_time(self):
        return self.operation_conversion_time

    def get_aod_encoding_translation_time(self):
        return self.aod_encoding_translation_time

    def __len__(self):
        return len(self.sequence)
    
    def __getitem__(self, key):
        return self.sequence[key]

    def pop(self, key):
        value = self.sequence.pop(key)
        return value


class AOD_Operation(object):
    '''
    Parent class of AOD operations.

    Attributes:
        aod_encoding : np.ndarray of size (2, N)
            represents action of x and y AOD channels on a grid during operation.
        
        type : str
            alpha or nu operation

    '''

    def __init__(self, aod_encoding, type):
        self.aod_encoding = aod_encoding
        self.type = type


class AlphaOperation(AOD_Operation):
    '''
    Alpha operation

    Attributes: 
    ----
        extract : bool
            whether extract or implant operation

        trap_indices : list
            list of static array trap indices for which alpha operation applies

    '''

    def __init__(self, aod_encoding: np.ndarray, trap_labels: list, extraction: bool):
        super().__init__(aod_encoding, type='alpha')
        self.extract = extraction
        self.trap_indices = trap_labels

    def apply(self, static_array, dynamic_array):
        if self.extract:
            for trap_idx in self.trap_indices:
                coords = static_array.get_trap_by_index(trap_idx).euclidean_coords
                dynamic_trap = dynamic_array.get_trap_by_coordinates(coords)
                dynamic_trap.set_on()
                static_array.extract_trap(trap_idx, dynamic_array)
        else:
            
            for trap_idx in self.trap_indices:
                dynamic_array.implant_trap(trap_idx, static_array)
                coords = static_array.get_trap_by_index(trap_idx).euclidean_coords
                dynamic_trap = dynamic_array.get_trap_by_coordinates(coords)
                dynamic_trap.set_off()

    def __str__(self):
        return str(self.aod_encoding)


class NuOperation(AOD_Operation):
    '''
    Alpha operation

    Attributes: 
    ----
        trap_movements : dict
            dict of from_trap -> to_trap describing which traps move during operation

    '''


    def __init__(self, aod_encoding, op: Operation):
        super().__init__(aod_encoding, type='nu')
        self.trap_movements = op.trap_movements

    def apply(self, dynamic_array):
        from_trap_atoms = {}


        # check if doing "red" or "rec" *HACK*
        first_move_from = list(self.trap_movements.keys())[0]
        first_move_to = self.trap_movements[first_move_from]
        #print(first_move_from, "First move to", first_move_to, "Block size:", len(list(self.trap_movements.keys()))) 
        if abs(first_move_to - first_move_from) > 1:
            is_red_not_rec = False
        else:
            is_red_not_rec = True

        for i, trap_idx in enumerate(self.trap_movements.keys()):
            #if(i > 0):
                #print(trap_idx, "to", self.trap_movements[trap_idx]) 
            #print("before moving:", dynamic_array.get_occupation_list())
            dynamic_trap = dynamic_array.get_trap_by_index(trap_idx)
            from_trap_atoms[self.trap_movements[trap_idx]] = dynamic_trap.clear_trap()
            # print("after clearing:", dynamic_array.get_occupation_list())
            if len(from_trap_atoms) > 0:
                dynamic_trap.undergo_nu_operation()
            if dynamic_trap.active and is_red_not_rec:
                dynamic_trap.set_off()
            
        for from_trap_idx in from_trap_atoms:
            atoms = from_trap_atoms[from_trap_idx]
            dynamic_trap = dynamic_array.get_trap_by_index(from_trap_idx)
            for atom in atoms:
                atom.undergo_nu_operation()
            if not dynamic_trap.active:
                dynamic_trap.set_on()
            dynamic_trap.trap_atom(atoms)
            # print("after trapping:", dynamic_array.get_occupation_list())

        

    def __str__(self):
        return str(self.aod_encoding)
