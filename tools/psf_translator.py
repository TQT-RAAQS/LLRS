import struct
import numpy as np
import sys
import os
from pathlib import Path
from experiment.toolkits.configs import Addresses

import pickle

#
# Return: file_path to destination to save psf file
#
def process_args(argv):
    num_args = len(argv)
    program = argv[0]
    file_path = -1

    help = f'''
    usage: python3 {program} <file path>
            
    notes: 
        <file path> = 'default' puts the psf files under LLRS/resources/psf
        <file path> should be entered as a relative path to the calling location. do not put a '/' at the end")
        the file path provided is not validated so make sure to enter it correctly!")
    '''

    if num_args > 1:    # two or more
        current_directory = os.getcwd()
        if argv[1] == '--help':
            print(help)
            exit()
        elif argv[1] == 'default':
            file_path = home + '/Experiment/experiment/modules/LLRS/resources/psf/'
        else:           # interperet as file path
            file_path = current_directory + '/' + argv[1] + '/'
    else:   # demand that args be included
        print(help)
        exit()

    return [file_path]

# 
# Writes a binary and txt psf file to the desired file path
# Return: 
# 
def generate_psf(file_path, params, binary=True):

    # psf_dict            = pickle.load(open(Addresses.traps_psf.replace("raaqs","raaqs3"), "rb"))
    
    psf_dict            = pickle.load(open(Addresses.traps_psf, "rb"))
#    psf_dict            = pickle.load(open("/home/tqtraaqs2/Z/Configs/2023-12-12/traps_psf.pickle", "rb"))
    centers             = psf_dict.get("centers")
    psf_values          = psf_dict.get("psfs")
    experiment_title    = psf_dict.get("experiment_title")
    box_size_w          = psf_dict.get("box_size_w")
    box_size_h          = psf_dict.get("box_size_h")
    cropping            = psf_dict.get("cropping")
    background          = psf_dict.get("background")

    num_atoms = len(centers)

    frame_bl_corner = (0,0)
    if cropping["is_cropping"] == True:
        cc = cropping["cropping_center"]
        cw = cropping["cropping_width"]
        ch = cropping["cropping_height"]
        frame_bl_corner = ( cc[0] - ch // 2  , cc[1] - cw // 2 )

    output_tuples = []
    for atom_num in range(num_atoms):
        # kernel_psf_values = psf_values[atom_num].flatten() # DKEA: verify this
        
        absolute_kernel_center_coords = frame_bl_corner + centers[atom_num]
        absolute_kernlel_bl_coords = absolute_kernel_center_coords - (box_size_w // 2, box_size_h // 2)

        for k_w in range(box_size_w):
            for k_h in range(box_size_h):
                pixel_num = k_w + k_h * box_size_w
                
                psf_val = psf_values[atom_num][k_h, k_w]

                # psf_val = kernel_psf_values[pixel_num]
                
                pixel_x = absolute_kernlel_bl_coords[0] + k_w
                pixel_y = absolute_kernlel_bl_coords[1] + k_h

                pixel_coord = ( pixel_x, pixel_y )

                flattened_pixel_coord = pixel_coord[0] + pixel_coord[1] * params["image_width"]

                output_tuples.append((atom_num, int(flattened_pixel_coord), psf_val))

    # print("output_tuples")
    # print(output_tuples)

    # serialize tuples
    file_name = os.path.join(file_path,"psfs.bin")

    if binary:
        with open(file_name, 'wb') as file:
            
            for (atom_idx, coord, psf_val) in output_tuples:
                atom_idx_binary = struct.pack("N", atom_idx)
                file.write(atom_idx_binary)

                idx_binary = struct.pack("N", coord)
                file.write(idx_binary)

                psf_value_binary = struct.pack("d", psf_val)
                file.write(psf_value_binary)

    return 

# 'Entry point' 
name = "1d_IT_1024_1024_21_traps"
psf_params_dict = {
    "name": name,
    "image_height": 1024,
    "image_width": 1024
}
home = os.path.expanduser("~")

[file_path] = process_args(sys.argv)
generate_psf(file_path, psf_params_dict, binary=True) 
print(f"psf files saved to {file_path}")
print("generate_psf() complete...")