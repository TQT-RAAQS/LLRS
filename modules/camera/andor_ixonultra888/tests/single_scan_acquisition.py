'''
Script to take an image in Run Till Abort acquisition mode
using an external trigger outputted by the AWG

@date: 2022-03-03
@author: Khush Bhavsar
@email: k4bhavsa@uwaterloo.ca
'''

from experiment.instruments.arbitrary_waveform_generator.spectrum_m4i66xx.spectrum_m4i66xx import *
from experiment.instruments.camera.basler_ace2.basler_ace2 import BaslerAce2
from experiment.instruments.camera.andor_ixonultra888.andor_ixonultra888 import AndoriXonUltra888
from experiment.toolkits.image_acquisition_and_processing.image_toolkit.image_acquisition import ImageAcquisition
# from experiment.modules.dynamic_trap_arrays.tools.waveform import Params, Waveform, sine, poly_sine
from experiment.modules.atom_sorting.aod_workstation.translator.waveform import Waveform, Static
from experiment.instruments.arbitrary_waveform_generator.spectrum_m4i66xx.spectrum_sequence import SequenceMode
from pypylon import pylon

import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
from PIL import Image

def save_images(nr_images):
    '''
    Runs a series of single scan acquisitions for 1000 images one after the other
    Collects data for runtimes
    '''  

    data_folder = '/home/tqtraaqs/Desktop/W2022-Data/EMCCD-Workstation/Dark Image Characterization'
    now = datetime.datetime.now()
    folder = os.path.join(data_folder, str(now)) # temp, gain, time
    os.mkdir(folder)

    # setting config path for camera
    config_path = {
        'emccd': {
            'static': '/home/tqtraaqs/Desktop/tqtraaq_git/Experiment/experiment/instruments/camera/andor_ixonultra888/andor_static_properties.yml',
            'dynamic': '/home/tqtraaqs/Desktop/tqtraaq_git/Experiment/experiment/instruments/camera/andor_ixonultra888/andor_acquisition_properties.yml'}
    }
    ia = ImageAcquisition({'emccd': AndoriXonUltra888()}, configuration_path=config_path)

    # opening connection to camera
    ia.open_connection()
    ia.set_static_properties()
    ia.set_acquisition_properties()
    emccd = ia._get_camera('emccd')
    emccd.image_buffer = np.zeros((emccd.xpixels, emccd.ypixels), dtype=np.int16)
    image_size = emccd.xpixels * emccd.ypixels
    print('Starting Acquisition')
    for i in range(nr_images):
        emccd.start_acquisition() #trigger acquisition on camera
        ret = emccd.wait_for_acquisition()
        (ret, full_frame_buffer, validfirst, validlast) = emccd.sdk.GetImages16(1, 1, image_size)
        emccd.image_buffer = np.array(full_frame_buffer, dtype=np.int16).reshape(
            emccd.xpixels, emccd.ypixels
        )
        im = Image.fromarray(emccd.image_buffer)
        filepath = os.path.join(folder, 'Image ' + str(i + 1) + '.png')
        im.save(filepath)
    print('Acquisition done')
    meta_info_filepath = folder + '/meta_information.yml'
    meta_info = emccd.meta_information
    with open(meta_info_filepath, 'w') as outfile:
        yaml.dump(meta_info, outfile, default_flow_style=False)
    ia.close_connection()
if __name__ == '__main__':
    save_images(100)



    


