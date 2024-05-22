'''
Helper functions to aid in the benchmarking module of the LLRS
Authors: Timur Khayrullin, Evan Yu Summer 2023
'''

import uuid
import json
import time
import re
import os
import pickle
import pandas as pd
import numpy as np
import pathlib
import pickle
import base64
from copy import deepcopy
from pathlib import Path
from structures.geometry import Params, ArrayGeometry
from utils.emulator_utils import LossParams, load_array
from models.array_model import StaticTrapArray, Coordinate
from controllers.reconfiguration_sequence import AlphaOperation, Operation, NuOperation

'''
Hardcoded loss parameters. This is here because otherwise they are 
hardcoded to the LossParams() constructor in the operational/utils/emulator_utils. 
These can be modified by benchmark_params.yml if found in the problem_definition section
'''
default_loss_atom_params  = { "p_alpha" : 0.98,
                        "p_nu" : 0.98,
                        "t_lifetime" : 60}

default_loss_env_params = { "t_alpha" : 10e-6,
                        "t_nu" : 10e-6,
                        "t_latency" : 0.02,}


'''
Hardcoded geometry parameters. This is here because otherwise they are 
hardcoded to the Params() constructor in operational/structures/geometry.py. 
These can be modified by benchmark_params.yml if found in the problem_definition section
'''
default_geometry_params = {
    "trap_number" : [],
    "norms" : 2 * 1e-6,
    "sampling_index" : [],
    "z" : 0,
    "shift_vector" : [0,0,0],
    "k1" : [1,0,0],
    "k2" : [0,1,0],
    "k3" : [0,0,1],
    "euler_angle" : [[0, 0, 0],"rad"],
    "coordinates" : [] 
}


'''
Converts dictionary to pp.Params() object from operational/structures/geometry.py.
Used because geometry logic within Reconfiguration uses pp.Params() throughout
and unless its set explicitly, it uses its own default constructor. 
this is a band-aid solution that avoids re-implementation of geometry code.

Spring 2023 goal: remove the need for this subroutine, we should just pass geometry params directly
'''
def convert_to_geometry_params_obj(dict):
    result = Params()
    result.trap_number = dict["trap_number"]
    result.norms = dict["norms"]
    result.sampling_index = dict["sampling_index"]
    result.z = dict["z"]
    result.shift_vector = dict["shift_vector"]
    result.k1 = dict["k1"]
    result.k2 = dict["k2"]
    result.k3 = dict["k3"]
    result.euler_angle = tuple(dict["euler_angle"])
    result.coordinates = dict["coordinates"]

    return result

'''
Converts dictionary to LossParams() object from operational/utils/emulator_utils.
Used because loss logic within Reconfiguration uses LossParams() throughout
and unless its set explicitly, it uses its own default constructor. 
this is a band-aid solution that avoids re-implementation of loss code.

Spring 2023 goal: remove the need for this subroutine, we should just pass loss params directly
'''
def convert_to_loss_params_obj(dict):
    ret = LossParams()
    ret.p_alpha = dict["p_alpha"]
    ret.p_nu = dict["p_nu"]
    ret.t_alpha = dict["t_alpha"]
    ret.t_nu = dict["t_nu"]
    ret.t_lifetime = dict["t_lifetime"]

    return ret


'''
Returns: emccd_dynamic_properties dictionary with updated ROI parameters
emccd_dynamic_properties: original dynamic properties
experiment_params: experiment_params section of problem_definition from benchmark_params.yml
'''
def update_emccd_dyn_prop(emccd_dynamic_properties, experiment_params):

    return_prop = deepcopy(emccd_dynamic_properties)

    return_prop["cropheight"] = experiment_params["roi_height"]
    return_prop["cropWidth"] = experiment_params["roi_width"]
    return_prop["cropleft"] = experiment_params["roi_x_offset"]
    return_prop["cropbottom"] = experiment_params["roi_y_offset"]

    return return_prop

'''
Returns: True if new roi settings are compatible with the old ones, False otherwise
'''
def emccd_dyn_prop_check_compatible(current_exp_params, new_exp_params):

    if current_exp_params['roi_width'] != new_exp_params['roi_width']:
        return False
    if current_exp_params['roi_height'] != new_exp_params['roi_height']:
        return False
    if current_exp_params['roi_x_offset'] != new_exp_params['roi_x_offset']:
        return False
    if current_exp_params['roi_y_offset'] != new_exp_params['roi_y_offset']:
        return False

    return True
'''
Mutates user_dict to include any keys in config_dict if they are not defined
user_dict: user-defined dictionary that is augmented
config_dict: system-defined dictionary used to augment the user dictionary 
'''
def augment_dict(user_dict, config_dict):

    for key in config_dict:
        if key not in user_dict:
            user_dict[key] = config_dict[key]

'''
changes problem_definition parameters according to an index. For the ith problem, 
replaces any given parameter with the one in problem_range_dict at the ith index in 
the list for that parameter. If no list is present, keeps the default.
NOTE: all lists defined in the problem_range section must have the same length

Returns: copy of base_problem_dict mutated to use the list[pblm_idx] values instead of the default, 
for all lists defined in the problem_range_dict

base_problem_dict: defaults for problem_definition
problem_range_dict: dict that mimics same structure as base_problem_dict but has lists instead of values
pblm_idx: index to use for each list
'''
def augment_problem_dict(base_problem_dict, problem_range_dict, pblm_idx):

    result_dict = deepcopy(base_problem_dict) # deepcopy dict

    for category in problem_range_dict:
        # first level is always a dict
        for var in problem_range_dict[category]:
            result_dict[category][var] = problem_range_dict[category][var][pblm_idx]
    

    # input validation goes here


    # if target config is a string, it implies a standard target config (ex "center_compact")
    # so we use a standard loading procedure
    if type(result_dict["problem_params"]["target_config"]) == str:
        result_dict["problem_params"]["target_config"] = \
            create_standard_target(result_dict["problem_params"]["target_config"],
                                  result_dict["problem_params"])


    result_dict["uuid"] = str(uuid.uuid4())

    return result_dict



'''
returns true if the given problem dict describes a 2D problem
'''
def is_2D_problem(problem_def_dict):

    Nt_x = problem_def_dict["problem_params"]["Nt_x"]
    Nt_y = problem_def_dict["problem_params"]["Nt_y"]

    return (Nt_x > 1 and Nt_y > 1)

'''
Makes a list of problem objects, given problem definitions. 
Problem objects are dictionaries that store initial and target StaticTrapArrays

Returns: a list of static trap array pairs, initial and target.

problem_def_dict: problem definition dictionary
num_trials: number of initial configurations to create
trial_selector: select which trial to run
'''
def make_problem_object(problem_def_dict, num_trials, allow_deficit, trial_selector):

    # segment problem_def dict for readability
    problem_params =  problem_def_dict["problem_params"]
    geometry_params = problem_def_dict["geometry_params"]
    loss_params = problem_def_dict["loss_atom_params"]

    # complete geometry
    geometry_params["trap_number"] = [problem_params["Nt_x"], problem_params["Nt_y"]] # TODO make prettier
    geometry = ArrayGeometry(problem_params["array_geometry_type"], convert_to_geometry_params_obj(geometry_params))

    # make static arrays
    empty_static_array = StaticTrapArray(geometry)
    target_static_array = StaticTrapArray(geometry)

    # fill target array
    load_array(target_static_array, occupation_indices=np.asarray(problem_params["target_config"]).nonzero()[0], 
               loss_params=loss_params)

    # make problem object
    result = []

    # for each trial, randomly fill initial array and add it to problem object
    for counter in range(num_trials):

        trial_static_array = deepcopy(empty_static_array)

        # when num_target is set to None, the initial static array won't always have num_target atoms
        num_target = problem_params["num_target"] if not allow_deficit else None

        load_array(trial_static_array, load_efficiency=problem_params["load_efficiency"],
                   loss_params=loss_params, num_target=num_target)
        if counter == trial_selector or trial_selector == -1:
            result.append({"initial" : trial_static_array,
                        "target" : target_static_array})
            
    
    return result


'''
Makes a problem single problem object, given a problem definition and an initial config list. 
Problem objects are dictionaries that store initial and target StaticTrapArrays

Returns: a list of static trap array pairs, initial and target.

problem_def_dict: problem definition dictionary
initial_config_list: list of initial configurations (binary arrays)
'''
def load_problem_object(problem_def_dict, initial_config_list):

    # segment problem_def dict for readability
    problem_params =  problem_def_dict["problem_params"]
    geometry_params = problem_def_dict["geometry_params"]
    loss_params = problem_def_dict["loss_atom_params"]

    # complete geometry
    geometry_params["trap_number"] = [problem_params["Nt_x"], problem_params["Nt_y"]] # TODO make prettier
    geometry = ArrayGeometry(problem_params["array_geometry_type"], convert_to_geometry_params_obj(geometry_params))

    # make static arrays
    empty_static_array = StaticTrapArray(geometry)
    target_static_array = StaticTrapArray(geometry)

    # fill target array
    load_array(target_static_array, occupation_indices=np.asarray(problem_params["target_config"]).nonzero()[0], 
               loss_params=loss_params)

    # make problem object
    result = []

    # for each trial, fill initial according to provided list of binary arrays
    for initial_config in initial_config_list:

        trial_static_array = deepcopy(empty_static_array)

        load_array(trial_static_array, occupation_indices=np.asarray(initial_config).nonzero()[0], 
               loss_params=loss_params)

        result.append({"initial" : trial_static_array,
                       "target" : target_static_array})
    
    return result

def config_to_b64(config):
    int_enc = int(''.join(str(bit) for bit in config), 2)
    int_bytes = int_enc.to_bytes((len(config) + 7) // 8, byteorder='big')
    return base64.b64encode(int_bytes).decode("utf-8")


'''
loads a StaticTrapArray with num_target atoms in the center

target_static_array: StaticTrapArray to be loaded (this object is mutated)
num_target: number of atoms to load in the center
loss_params: loss parameters to initialize atoms with
'''
def create_centered_config(num_trap, num_target):

    result = [0] * num_trap

    start_index = (num_trap//2) - (num_target//2) # left side of line of atoms

    for offset in range(num_target):

        result[start_index+offset] = 1

    return result

def create_standard_target(target_desc, problem_params):

    if(target_desc == "center compact"):

        return create_centered_config(problem_params["Nt_x"] * problem_params["Nt_y"], problem_params["num_target"])
    
    else:
        Exception("standard target not supported")


'''
saves all benchmarking settings and complete problem definitions to individual json files per problem.
jsons are named according to their uuid.

problem_dict_list: problem definition dictionaries
benchmark_params: all settings in benchmark_params.py
reconfig_problems: list of problem objects
problems_folder: path to save problem to
'''
def save_full_problem_definitions(problem_dict_list, benchmark_params, reconfig_problems, problems_folder):

    for (problem_dict, problem_obj) in zip(problem_dict_list, reconfig_problems):

        result_json = deepcopy(benchmark_params)

        # remove problem range section from 

        # save all problem parameters
        result_json["problem_definition"] = problem_dict

        problem_path = f"{problems_folder}/{problem_dict['uuid']}.json"

        with open(problem_path, "w+") as file:
            json.dump(result_json, file, indent=4)


'''
save every problem's solution to it's own json, named according to the problem's uuid.
the json format labels each trial, repetition and cycle

problem_dict_list: problem definition dictionaries
obp_sols: list of solutions for all problems, this is a 4d array with dimensions
    indexed by: [problem num][trial num][repetition num][cycle num][atom array]
solutions_path: path to save solution to
'''
def save_obp_sols(problem_def_dict_list, obp_sols, solutions_path):

    for i, (problem, sol) in enumerate(zip(problem_def_dict_list, obp_sols)):

        uuid = problem["uuid"]

        result_dict = {
            f"trial_{i}": {
                
                f"repetition_{j}":{
                    
                    f"cycle_{k}": sol[i][j][k]
                    for k in range(len(sol[i][j]))
                }
                for j in range(len(sol[i]))
            }
            for i in range(len(sol))
        }

        sol_path = f'{solutions_path}/{uuid}.json'

        with open(sol_path, "w+", encoding='utf-8') as json_file:
            json.dump(result_dict, json_file, indent=4)

'''
Serialize a single aod operation. 
Used to save aod_operations in an operational benchmarking session
'''
def serialize_aod_op (op):
    output = {"type": op.type}
    if op.type == "alpha":
        output["trap_indices"] = ",".join(map(str, op.trap_indices))
        output["extract"] = op.extract
    else:
        output["trap_movements_src"] = ",".join(map(str, op.trap_movements.keys()))
        output["trap_movements_dst"] = ",".join(map(str, op.trap_movements.values()))
    return output

'''
Deserialize a single aod operation. 
Used to load saved aod moves from an operational benchmarking session
'''
def deserialize_aod_op(aod_op_raw):
    if aod_op_raw["type"] == "alpha":
        trap_indices = list(map(int, aod_op_raw["trap_indices"].split(",")))    
        return AlphaOperation(None, trap_indices, aod_op_raw["extract"])
    else:
        trap_movements_src = list(map(int, aod_op_raw["trap_movements_src"].split(",")))
        trap_movements_dst = list(map(int, aod_op_raw["trap_movements_dst"].split(",")))
        return NuOperation(None, Operation(zip(trap_movements_src, trap_movements_dst)))

'''
Helper function for store_formatted_benchmark_data
Processes runtime and operational benchmarking data and turns them into a dataframe
'''    
def remove_prefix(string):
        match = re.search(r'\d+', string)
        if match:
            return int(match.group()) 
        return None


'''
Helper function that takes a set of jsons and compiles them into a Pandas dataframe
'''
def create_dataframe_from_json(runtime_data, config_data, metrics_data, aod_ops_data, is_2D):

    df = pd.DataFrame.from_dict({
        (remove_prefix(i), remove_prefix(j), remove_prefix(k)): 
            { 
                **(runtime_data[i][j].get(k, {}) if runtime_data else {}), 
                **metrics_data[i][j].get(k, {}),
                **{
                    "aod_operations": aod_ops_data[i][j].get(k, None),
                    "config": config_data[i][j].get(k, None)
                }
            }
            for i in metrics_data.keys()
            for j in metrics_data[i].keys()
            for k in metrics_data[i][j].keys()
        },
        orient='index')
    
    df = df.reset_index()

    # Rename the columns
    rename_dict = {
        df.columns[0]: "Trial",
        df.columns[1]: "Repetition",
        df.columns[2]: "Cycle"
    }

    df.rename(columns=rename_dict, inplace=True)

    return df

'''
New format function to replace the old formatter which is some
'''
def store_formatted_benchmark_data(experiment_id, runtime_dir, metrics_dir, config_dir, aod_ops_dir, problem_def_dir, output_path, is_2D):

    runtime_path = Path(runtime_dir) / Path(experiment_id + ".json")
    metrics_path = Path(metrics_dir) / Path(experiment_id + ".json")
    config_path = Path(config_dir) / Path(experiment_id + ".json")
    aod_ops_path = Path(aod_ops_dir) / Path(experiment_id + ".json")
    problem_def_path = Path(problem_def_dir) / Path(experiment_id + ".json")

    if not os.path.exists(metrics_path) or \
        not os.path.exists(config_path) or \
        not os.path.exists(aod_ops_path) or \
        not os.path.exists(problem_def_path):
        return
    
    runtime_data = None
    if os.path.exists(runtime_path):
        with open(runtime_path, 'r') as file1:
            runtime_data = json.load(file1)

    with open(config_path, 'r') as file2:
        config_data = json.load(file2)

    with open(metrics_path, 'r') as file3:
        metrics_data = json.load(file3)

    with open(aod_ops_path, 'r') as file4:
        aod_ops_data = json.load(file4)

    with open(problem_def_path, 'r') as file5:
        problem = json.load(file5)

    problem["data"] = create_dataframe_from_json(runtime_data, config_data, metrics_data, aod_ops_data, is_2D)
    
    with open(output_path, 'wb') as file:
        pickle.dump(problem, file, protocol=pickle.HIGHEST_PROTOCOL)