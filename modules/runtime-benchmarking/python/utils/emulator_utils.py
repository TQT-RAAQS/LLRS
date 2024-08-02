'''
Authors: Unknown (pre 2023), Wendy Lu (Winter 2023)
'''

from typing import List, Union, Tuple, Set
from pprint import pprint

import numpy as np
from numba import jit
import params as pp

from numba import float32
from numba.experimental import jitclass
from structures.geometry import Params, ArrayGeometry
from controllers.reconfiguration_sequence import Operation, AlphaOperation, NuOperation
from models.array_model import StaticTrapArray
from scipy.stats import bernoulli, binom


class LossParams(object):
    def __init__(self, 
                 p_alpha=0.985, 
                 p_nu=0.985,
                 t_alpha=10*1e-6,
                 t_nu=10 * 1e-6,
                 t_latency=0.00,
                 t_lifetime = 60):
        
        self.p_alpha = p_alpha
        self.p_nu = p_nu
        self.t_alpha = t_alpha
        self.t_nu = t_nu
        self.t_latency = t_latency
        self.t_lifetime = t_lifetime

    def copy(self):
        return LossParams(self.p_alpha, 
                          self.p_nu,
                          self.t_alpha,
                          self.t_nu,
                          self.t_latency,
                          self.t_lifetime)

'''
Calculates (projects) which atoms got lost based on the atoms corruption (based on LossParams)
static_array: StaticTrapArray onto which we project state
sequence_time: elapsed time of the reconfiguration
'''
def project_state(static_array, sequence_time):
    kept = 0
    atoms_lost_indices = []
    for trap_idx in static_array.array_object_by_idx.keys():
        trap = static_array.get_trap_by_index(trap_idx)
        if len(trap.atoms) > 0:
            if np.random.random() < list(trap.atoms)[0].get_corruption(sequence_time):
                kept += 1
                list(trap.atoms)[0].reset_corruption()
            else:
                atoms = trap.clear_trap()
                for atom in atoms:
                    atoms_lost_indices.append(atom.get_id())
    
    return atoms_lost_indices

def problem_solvable(initial_array, target_array):
    if len(initial_array.get_occupation_state()) >= len(target_array.get_occupation_state()):
        return True
    return False

def target_state_met(initial_array, target_array):
    if set(target_array.get_occupation_state()).issubset(set(initial_array.get_occupation_state())):
        return True
    return False


def create_random_occ_state(num_trap, load_efficiency, num_target = None):
    if num_target is not None:
        def truncated_binom(n, p, min_value):
            probs = binom.pmf(np.arange(min_value, n+1), n, p)
            probs /= probs.sum()
            return np.random.choice(np.arange(min_value, n+1), p=probs)

        num_filled = truncated_binom(num_trap, load_efficiency, num_target)
        occupation_status = np.append(np.ones(num_filled), np.zeros(num_trap - num_filled))
        np.random.shuffle(occupation_status)
        return occupation_status
    return bernoulli.rvs(load_efficiency, size = num_trap)


'''
Loads a StaticTrapArray with atoms according to load_efficiency or num_atom or occupation indices

static_array: StaticTrapArray
load_from_indices: boolean, whether load by indices of atoms
load_efficiency: float, (0,1)
occupation_indices: list of indices, from 0 to num_trap - 1
'''
def load_array(static_array, num_atom = None, load_efficiency = None, 
               occupation_indices = None, loss_params = None, num_target=None):
    
    assert(num_atom or load_efficiency or occupation_indices is not None)
    assert(not((num_atom and load_efficiency) or (num_atom and occupation_indices) 
               or (load_efficiency and occupation_indices)))

    if num_atom:
        static_array.set_occupation_state(filling_type = 'random', n_atoms = num_atom, loss_params = loss_params)

    elif load_efficiency:
        num_trap = static_array.trap_number

        # this is a binary array representing a loaded array
        occupation_status = create_random_occ_state(num_trap, load_efficiency, num_target)

        occupation_indices = np.where(occupation_status != 0)[0]

        static_array.set_occupation_state(filling_type = 'indices', trap_indices = occupation_indices, loss_params = loss_params)

    elif occupation_indices is not None:
        static_array.set_occupation_state(filling_type = 'indices', trap_indices = occupation_indices, loss_params = loss_params)

    else:
        Exception('Not enough parameters provided.')

'''
Returns : a binary list containing 1 at the indices provided

*this method should be a member of StaticTrapArray*
'''
def translate(index_list, num_trap):
    arr = [0] * num_trap
    for idx in index_list:
        arr[idx] = 1
    return arr



def solver_output_to_aod_ops(algorithm, trap_array, solver_output):
    Nt_x, Nt_y, Nt_z = trap_array.trap_number_along_generators
    aod_ops = []
    op_size = 4 
    raw_ops = [solver_output[i:i+op_size] for i in range(0, len(solver_output), op_size)]
    dir_dict = {3:1, 4:1, 8: 1 ,9: 1, 10: Nt_x, 11: Nt_x }
    for raw_op in raw_ops:
        move_type, index, offset, block_size = raw_op 

        if move_type in [1, 2, 6, 7]:  # Extract/Implant
            traps = [offset + i*Nt_x for i in range (index, index+block_size)]
            # print(traps)
            aod_ops.append(AlphaOperation([], traps, move_type==2 or move_type==7))    
        elif move_type != 0:
            dir = dir_dict[move_type]
            if move_type in [3, 8, 10]:   # Right/UP 
                moves = [(offset + i*Nt_x, offset + i*Nt_x + dir)
                        for i in range (index, index+block_size)]
            else:                         # Left/Down
                moves = [(offset + i*Nt_x + dir, offset + i*Nt_x)
                        for i in range (index, index+block_size)]
            # print(moves)
            aod_ops.append(NuOperation([], Operation(moves)))
        
    return aod_ops

'''
static_array: StaticTrapArray
dynamic_array: DynamicTrapArraySpace
operations: list of AOD_Operation

Return:
operation_time: time for applying all the aod operations, determined by loss_params
'''
def apply_aod_operation(static_array, dynamic_array, operations, t_alpha, t_nu):
    operation_time = 0
    
    # debug
    intermediate_configs = []


    for op in operations:
        
        # THIS METHOD HAS DEBUG STATEMENTS THROUGHOUT, NEEDS CLEANING AFTER VALIDATION
        moves_info = None
        if op.type == 'nu':
            moves_info = op.trap_movements
            # print("Trap Movements:", moves_info)
            op.apply(dynamic_array)
            operation_time += t_nu
            # static_array.print_trap_contents()
        elif op.type == 'alpha':
            moves_info = (f"Extracting: {op.extract}", op.trap_indices)
            # print(moves_info)
            op.apply(static_array, dynamic_array)
            operation_time += t_alpha
            # if (op.extract == False):
                # static_array.print_trap_contents()


    
        # intermediate_configs.append(
        #     set(static_array.get_occupation_state() + dynamic_array.get_occupation_state()))
    
    return operation_time, intermediate_configs
