#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: Barry Cimring
'''
import numpy as np
from structures.graph import DynamicArrayGraph
from controllers.reconfiguration_sequence import Operation, AlphaOperation, NuOperation, SubSequence
from copy import deepcopy

def generate_operation_on_2d_grid(cols: list, rows: list, displace_in_rows: bool, right_down: bool, shape: tuple):
    traps_displaced_to = []
    for col in cols:
        for row in rows:
            if displace_in_rows:
                trap_label = shape[1]*(shape[0] - 1 - row) + col
                destination_trap_label = trap_label + right_down[col]
            else:
                trap_label = shape[1]*(shape[0] - 1 - row) + col
                destination_trap_label = trap_label - shape[1]*(right_down[row])
            
            traps_displaced_to.append((trap_label, destination_trap_label))
    
    return traps_displaced_to


def generate_alpha_substep_on_2d_grid(cols: list, rows: list, shape: tuple, extract: bool):
    trap_labels = []

    for col in cols:
        for row in rows:
            trap_label = shape[1]*row + col
            trap_labels.append(trap_label)
    
    return AlphaOperation(trap_labels, extraction=extract)


def generate_nu_substep_on_2d_grid(cols: list, rows: list, displace_in_rows: bool, forward_up: bool, shape: tuple):
    trap_displaced_to = {}

    for col in cols:
        for row in rows:
            trap_label = shape[1]*row + col
            if displace_in_rows:
                destination_trap_label = trap_label + 1*(-1 + 2*forward_up)
            else:
                destination_trap_label = trap_label + shape[1]*(-1 + 2*forward_up)
            
            trap_displaced_to[trap_label] = destination_trap_label
    
    return NuOperation(trap_displaced_to)

def get_operations(op_list):
    '''
    Convert list of abstract displacements on a graph to a list of Operation objects. 

    Args:
        op_list (List[dict]) : list of dicts of from_trap -> to_trap

    Returns:
        List[Operation]

    '''

    operation_obj_list = []
    for op in op_list:
        #print(op)
        operation_obj_list.append(Operation(op))

    return operation_obj_list

def verify_sequence(sequence: SubSequence, graph: DynamicArrayGraph, initial_occupation_state: list, target_occupation_state: list):
    '''
    Verifys if a solution to the reconfiguration problem defined on 
    a graph with initial and target occupation state is valid.

    Args:
        sequence (SubSequence) : sequence of operations encoding solution
        graph (DynamicArrayGraph) : graph representing set of vertices and possible displacements
        initial_occupation_state (list) : list of vertices initially occupied
        target_occupation_state (list) : list of vertices occupied in the target state

    Returns
        bool

    Throws:
        Exception if 
            (1) any traps multiply occupied at any time
            (2) state not a subset of vertices at any time
            (3) initial state not a subset of target state
            (4) any displacement in sequence not along an edge in graph
    '''
    
    #print("initial: ", initial_occupation_state)
    #print("target: ", target_occupation_state)
    tmp_state = deepcopy(initial_occupation_state)

    if tmp_state.sort() != list(set(tmp_state)).sort():
        Exception('Multiply populated traps')
    V = graph.get_all_vertices()

    assert(set(initial_occupation_state).issubset(V))
    
    for op in sequence.sequence:
        for from_trap in op.trap_movements.keys():
            to_trap = op.trap_movements[from_trap]
            if to_trap in tmp_state:
                raise Exception('Atom in trap ' + str(from_trap) + ' displaced to trap ' + str(to_trap) + ' already occupied')
            if (to_trap not in V) or (from_trap not in V):
                raise Exception('Traps displaced not in graph vertices')
            if to_trap not in graph.get_edges(from_trap):
                raise Exception('Traps displaced not along graph edges')
            tmp_state[tmp_state.index(from_trap)] = to_trap

    if not set(target_occupation_state).issubset(set(tmp_state)):
        raise Exception('Target state not a subset of final state')

    return True