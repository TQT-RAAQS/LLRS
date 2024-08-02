import struct
import numpy as np
import sys
import os
import pickle 
from experiment.toolkits.configs import Addresses
import subprocess



def get_gaussian(kernal_height, kernal_width, sigma_x, sigma_y, x_center=0.0, y_center=0.0):
    x = np.linspace(-kernal_width/2, kernal_width/2, kernal_width)
    y = np.linspace(-kernal_height/2, kernal_height/2, kernal_height)
    xx, yy = np.meshgrid(x, y)

    gaussian = np.exp(-((xx-x_center)**2/(2*sigma_x**2) +
                      (yy-y_center)**2/(2*sigma_y**2)))
    sum = gaussian.sum()
    gaussian /= sum

    return gaussian

def get_psf_pickle_dictionary(Nt, kernal):
    psf_data = {}

    psfs = []
    centers = []
    x0, y0 = kernal//2 + kernal%2 - 1, kernal//2 +kernal%2 - 1 
    for i in range(Nt):
        x = i % Nt
        y = i // Nt
        centers.append(np.array([x0 + x*kernal, y0 + y*kernal]))
        psfs.append(get_gaussian(kernal, kernal, kernal/2, kernal/2))
    cropping = {"is_cropping": False}

    psf_data = {"centers": centers, "psfs": psfs, "cropping": cropping, "box_size_w": kernal, "box_size_h": kernal, "Nt": Nt}
    return psf_data

if (len(sys.argv) < 4):
    print("Usage: python generate-fake-psf.py <Nt> <kernal> <output_file>")
    sys.exit(1)

psf_data = get_psf_pickle_dictionary(int(sys.argv[1]), int(sys.argv[2]))
with open (sys.argv[3], "wb") as file:
    pickle.dump(psf_data, file)

