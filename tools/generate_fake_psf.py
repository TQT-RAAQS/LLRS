import struct
import numpy as np
import sys
import os

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
            file_path = current_directory + '/../resources/psf/'
        else:           # interperet as file path
            file_path = current_directory + '/' + argv[1] + '/'
    else:   # demand that args be included
        print(help)
        exit()

    return [file_path]

# 
# Return: 1d np array of density of 2d gaussian 
# 
def get_gaussian(kernel_height, kernel_width, sigma_x, sigma_y, x_center=0.0, y_center=0.0):
    x = np.linspace(-kernel_width/2, kernel_width/2, kernel_width)
    y = np.linspace(-kernel_height/2, kernel_height/2, kernel_height)
    xx, yy = np.meshgrid(x, y)

    gaussian = np.exp(-((xx-x_center)**2/(2*sigma_x**2) +
                      (yy-y_center)**2/(2*sigma_y**2)))
    sum = gaussian.sum()
    gaussian /= sum
    gaussian.shape = (kernel_width * kernel_height,)

    return gaussian

# 
# Writes a binary and txt psf file to the desired file path
# Return: 
# 
def generate_psf(file_path, params, print_kernels=True, rotate=True, binary=True):

    # flip dimensions if we're rotating
    if rotate:
        params["num_trap_x"], params["num_trap_y"] = \
        params["num_trap_y"], params["num_trap_x"] 

    # get psf functions (discretized guassian kernels) according to the array of sigma_x and sigma_y
    psfs = []
    for [sigma_x, sigma_y] in params["sigma_pair_arr"]:
        psf = get_gaussian(params["kernel_height"], params["kernel_width"], sigma_x, sigma_y)
        psfs.append(psf)

    # calculate top-left coord of each kernel
    kernel_corners = []
    for row_idx in range(params["num_trap_y"]):

        for column_idx in range(params["num_trap_x"]):

            pos = params["trap_start_index"] + column_idx * params["delta_x"] + row_idx * params["delta_y"] * params["image_width"]
            
            kernel_corners.append(pos)
    
    # get coordinates of all kernels, in a nested list
    kernel_coords = []
    for corner in kernel_corners:
        
        # stores coords for individual kernels
        kernel_temp = []

        for k_y in range(params["kernel_height"]):

            for k_x in range(params["kernel_width"]):

                kernel_temp.append(corner + k_x + k_y * params["image_width"])
        
        kernel_coords.append(kernel_temp)


    # create indices array, rotate indices if needed
    atom_indices = np.arange(params["num_trap_x"] * params["num_trap_y"])
    translated_indices = atom_indices
    if rotate:
        translated_indices = atom_indices.reshape(params["num_trap_x"], params["num_trap_y"])
        translated_indices = np.rot90(translated_indices, 1)
        translated_indices = translated_indices.flatten()

    # collect info into tuples
    output_tuples = []
    for atom_idx in atom_indices:

        for (coord, psf_val) in zip(kernel_coords[atom_idx], psfs[atom_idx]):

            output_tuples.append((translated_indices[atom_idx], coord, psf_val))

    for (atom, coord, psf_val) in output_tuples:
        print("atom: ", atom)
        print("coord: ", coord)
        print("psf_val: ", psf_val)
        print()

    return

    # serialize tuples
    file_name = file_path + params["name"] + f".{params['num_trap_x']}_{params['num_trap_y']}"

    if binary:
        with open(file_name+".bin", 'wb') as file:
            
            for (atom_idx, coord, psf_val) in output_tuples:

                atom_idx_binary = struct.pack("N", atom_idx)
                file.write(atom_idx_binary)

                idx_binary = struct.pack("N", coord)
                file.write(idx_binary)

                psf_value_binary = struct.pack("d", psf_val)
                file.write(psf_value_binary)
    else:
        with open(file_name+".txt", 'w') as file:
        
            for (atom_idx, coord, psf_val) in output_tuples:

                file.write(str(atom_idx) + " ")
                file.write(str(coord) + " ")
                file.write(str(psf_val) + " ")
                file.write("\n")

    # print binary grid
    if print_kernels:

        output_grid = ['_.'] * (params["image_height"] * params["image_width"])

        for kernel in kernel_coords:
            for coord in kernel:
                output_grid[coord] = "X."

        for atom_idx, corner in zip(translated_indices, kernel_corners):
            for i, char in enumerate(str(atom_idx)):
                output_grid[corner+i] = char + "."

        with open(file_name+".KERNEL_DISPLAY.txt", 'w') as file:
            for i in range(len(output_grid)):

                file.write(output_grid[i])
                if (i+1)%params["image_width"] == 0:
                    file.write("\n")

# 'Entry point' 

num_traps_x = 25
num_traps_y = 1
name = "1d_IT_1024_1024_11_atoms"
psf_params_dict = {
    "name": name,
    "sigma_pair_arr": np.ones((num_traps_x * num_traps_y,2)),
    "kernel_height":5,
    "kernel_width":5,
    "delta_x":10,
    "delta_y":10,
    "num_trap_x": num_traps_x,
    "num_trap_y": num_traps_y,
    "trap_start_index":2050,
    "image_height": 1024,
    "image_width": 1024
}

[file_path] = process_args(sys.argv)
generate_psf(file_path, psf_params_dict, print_kernels=True, rotate=True, binary=True) 
print(f"psf files saved to {file_path}")
print("generate_psf() complete...")
