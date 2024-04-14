'''
Author: Wendy Lu, Timur Khayrullin Winter 2023
'''

import os
import pathlib
import json
import uuid
import time
import datetime
import numpy as np
import logging
from copy import deepcopy
from utils.emulator_utils import project_state, target_state_met, \
problem_solvable, apply_aod_operation, solver_output_to_aod_ops
from models.array_model import StaticTrapArray, DynamicTrapArraySpace
from structures.geometry import ArrayGeometry
from solver import execute_wrapper
from benchmark_helpers import convert_to_geometry_params_obj, serialize_aod_op

#! We assume the algorithm is 1D solver
class OperationalBenchmarkingProblem(object):
    
    '''
    A module capable of performing Monte-Carlo simulation on pre-defined initial static arrays, 
    target static arrays, and algorithm.
    '''
    def __init__(self, initial_static_trap: StaticTrapArray, target_static_trap: StaticTrapArray, 
                  num_repetitions: int, algorithm: str):
        '''
        Initializes OperationalBenchmarkingProblem object.

        Args:
            initial_static_trap: initial atom configuration
            target_static_trap: target atom configuration
            geometry: trap geometry
            algorithm: algorithm for the reconfiguration problem
        '''
        self.simulated_static_trap = initial_static_trap
        self.initial_static_trap_copy = deepcopy(initial_static_trap)
        self.target_static_trap = target_static_trap

        self.dynamic_trap = DynamicTrapArraySpace(self.simulated_static_trap.geometry)
        self.num_repetitions = num_repetitions
        self.algorithm = algorithm

        self.num_traps = self.simulated_static_trap.trap_number
        self.Nt_x = self.simulated_static_trap.trap_number_along_generators[0]
        self.Nt_y = self.simulated_static_trap.trap_number_along_generators[1]
        if  self.Nt_x > 1 and \
            self.Nt_y > 1:
            
            self.dimension = 2
        else:

            self.dimension = 1

    def get_num_atoms (self, aod_ops):
        return sum(self.simulated_static_trap.get_occupation_list().tolist())
    
    def get_nu_ops (self, aod_ops):
        cnt = 0
        for op in aod_ops:
            cnt += int(op.type == "nu")
        return cnt

    def get_alpha_ops (self, aod_ops):
        cnt = 0
        for op in aod_ops:
            cnt += int(op.type == "alpha")
        return cnt

    # temporary metric for debugging
    def get_ele_nu_ops (self, aod_ops):
        cnt = 0
        for op in aod_ops:
            if op.type == "nu":
                cnt += len(op.trap_movements)
        return cnt

    # temporary metric for debugging
    def get_ele_alpha_ops (self, aod_ops):
        cnt = 0
        for op in aod_ops:
            if op.type == "alpha":
                cnt += len(op.trap_indices)
        return cnt

    
    def get_alpha_ops_atoms (self, aod_ops):
        result = []
        trap_array = self.simulated_static_trap.array_object_by_idx
        for trap in trap_array.values():
            result.append(sum(atom.N_alpha for atom in trap.atoms))
        return result
    
    
    def get_nu_ops_atoms (self, aod_ops):
        result = []
        trap_array = self.simulated_static_trap.array_object_by_idx
        for trap in trap_array.values():
            result.append(sum(atom.N_nu for atom in trap.atoms))
        return result
    
    def get_high_corruption_target_atoms (self, aod_ops):
        cnt = 0
        trap_array = self.simulated_static_trap.array_object_by_idx
        target_trap_array = self.target_static_trap.array_object_by_idx
        for trap, target_trap in zip(trap_array.values(), target_trap_array.values()):
            if (len(target_trap.atoms) > 0):
                cnt += sum(int(atom.get_corruption(0) < 0.95)  for atom in trap.atoms)
        return cnt
    
    def get_metrics (self, metric_functions, aod_ops):
        metrics = {}
        for metric_name in metric_functions.keys():
            metrics[metric_name] = metric_functions[metric_name](aod_ops)
        return metrics


    def pre_solve(self, loss, t_alpha, t_nu, latency, solver_wrapper_so_file):
        '''
        pre-solve operational_benchmarking_problem num_repetitions times, obtain the configuration of each cycle

        Returns:
        List(List(List))

        e.g.[[[1,1,0,0], [0,1,1,0]], [[1,1,0,0],[0,1,0,0]]] => num_repetitions = 2
        ''' 
        preloss_metrics = {
            "num_elementary_alpha_operations_atoms": self.get_alpha_ops_atoms,
            "num_elementary_nu_operations_atoms": self.get_nu_ops_atoms,
            "num_high_corruption_target_atoms": self.get_high_corruption_target_atoms
        }
        postloss_metrics = {
            "num_atoms": self.get_num_atoms,
            "num_nu_operations": self.get_nu_ops,
            "num_elementary_nu_operations": self.get_ele_nu_ops,
            "num_alpha_operations": self.get_alpha_ops,
            "num_elementary_alpha_operations": self.get_ele_alpha_ops,
        }
        
        target_binary_array = self.target_static_trap.get_occupation_list()

        configuration_array = [] 
        metrics_array = []
        aod_ops_array = []

        for rep_idx in range(self.num_repetitions):
            
            # if there is no loss, then each repetition is the same
            # we should try to decouple operational benchmarking & runtime benchmarking
            # so that we don't need *num_repetitions* copies of the same output
            if not loss and rep_idx >= 1:
                configuration_array.append(configuration_array[rep_idx-1])
                metrics_array.append(metrics_array[rep_idx-1])
                aod_ops_array.append(aod_ops_array[rep_idx-1])
                continue

            self.simulated_static_trap = deepcopy(self.initial_static_trap_copy)

            rep_config = [self.simulated_static_trap.get_occupation_list().tolist()]
            # placeholders for ease of processing
            rep_metrics = [self.get_metrics({**preloss_metrics, **postloss_metrics}, [])]
            rep_aod_ops = [[]]

            while not target_state_met(self.simulated_static_trap,
                                        self.target_static_trap):
                
                # for i in range(self.Nt_y):
                #     print(' '.join(map(str, self.simulated_static_trap.get_occupation_list().tolist()[i * self.Nt_x:(i + 1) * self.Nt_x])))
                
                if not problem_solvable(self.simulated_static_trap,
                                        self.target_static_trap):
                    break
                
                simulated_binary_array = self.simulated_static_trap.get_occupation_list()

                # run the corresponding algorithm and obtain batches(block operations)
                # src, dst, block_size, batch_ptr = solver.execute(simulated_binary_array, target_binary_array,
                #                                                 self.Nt_x, self.Nt_y) 
                solver_output = execute_wrapper(self.algorithm, simulated_binary_array, target_binary_array,
                                                       self.Nt_x, self.Nt_y, solver_wrapper_so_file)
                # print("Solver Output:")
                # for i in range(0, len(solver_output), 5):
                #     print(solver_output[i:i+5])

                # obtain list of aod operations(block operations)s
                aod_ops = solver_output_to_aod_ops(self.algorithm, self.simulated_static_trap, solver_output)
                # aod_ops = alg_output_to_aod_ops(self.algorithm, self.simulated_static_trap, src, dst, 
                #                                 block_size, batch_ptr)
                
    
                operational_time, intermediate_configs = apply_aod_operation(
                    self.simulated_static_trap, 
                    self.dynamic_trap, 
                    operations = aod_ops, 
                    t_alpha=t_alpha,
                    t_nu=t_nu
                )
        
                operational_time += latency

                # print(intermediate_configs)

                # some important metrics (e.g. alpha operations performed on each atom) get lost after projecting loss
                # so we must get the metrics first
                metrics = self.get_metrics(preloss_metrics, aod_ops)

                # determine the presence or absence of atoms based on possibility
                if loss:
                    project_state(self.simulated_static_trap, operational_time)

                metrics.update(self.get_metrics(postloss_metrics, aod_ops))

                # get the final configuration and update configuration array
                cycle_config = self.simulated_static_trap.get_occupation_list().tolist()
                rep_config.append(cycle_config)
                rep_metrics.append(metrics)
                rep_aod_ops.append(list(map(serialize_aod_op, aod_ops)))
            
            configuration_array.append(rep_config)
            metrics_array.append(rep_metrics)
            aod_ops_array.append(rep_aod_ops)

            # print("FINAL CONFIGURATION")
            # for i in range(self.Nt_y):
            #         print(' '.join(map(str, self.simulated_static_trap.get_occupation_list().tolist()[i * self.Nt_x:(i + 1) * self.Nt_x])))

        return configuration_array, metrics_array, aod_ops_array

