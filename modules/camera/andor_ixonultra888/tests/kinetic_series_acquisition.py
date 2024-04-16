'''
Example script performs kinetic series aquisition from the Andor iXon Ultra 888

Sailesh Bechar
Fall 2020

Brooke Dolny
Winter 2021
'''
from datetime import datetime
from experiment.instruments.camera.andor_ixonultra888.andor_ixonultra888 import AndoriXonUltra888
from experiment.instruments.camera.andor_ixonultra888.emccd_meta_information import EmccdMetaInformation


"""
full kinetic series acquisition workflow from the Andor iXon Ultra 888 camera


Args

----------

metaInformation (EMCCDMetaInformation): meta information of static and acquisition parameters of camera

Returns:

-------

image_buffer (numpy array): dimensions[xpixels, ypixels, num_images]

"""
EXPOSURE_TIME = 0.0001 # 0.0001 seconds is too short for actual acquisition
PREAMP_GAIN_INDEX = 1 # 0 - Gain 1.0, 1 - Gain 2.0
HS_SPEED_INDEX = 3
VS_SPEED_INDEX = 3
EM_GAIN = 500
EM_ADVANCED = 1 # 1 - Enable access to higher levels of EM gain (above 300)
EM_GAIN_MODE = 3
ACQUISITION_MODE = 3 # 3 : Kinetic Series
NUM_ACCUMULATIONS = 1
READOUT_MODE = 4 # 4 : Image
NUM_KINETICS = 50
KINETIC_CYCLE_TIME = 0.1 # time in seconds per each acquisition
DESIRED_TEMP = -95
TRIGGER_MODE = 0 # 0 : Internal
AMP_INDEX = 0 # 0 - EM, 1 - Conventional
BASELINE_CLAMP = 1
CAMERALINK_MODE = 1
FRAMETRANSFER_MODE = 1
MAX_PATH = 256

metaInformation = EmccdMetaInformation(
						EXPOSURE_TIME,
						PREAMP_GAIN_INDEX,
						HS_SPEED_INDEX,
						VS_SPEED_INDEX,
						EM_GAIN,
						EM_ADVANCED,
						EM_GAIN_MODE,
						ACQUISITION_MODE,
						NUM_ACCUMULATIONS,
						READOUT_MODE,
						NUM_KINETICS,
						KINETIC_CYCLE_TIME,
						DESIRED_TEMP,
						TRIGGER_MODE,
						AMP_INDEX,
						BASELINE_CLAMP,
						CAMERALINK_MODE,
						FRAMETRANSFER_MODE,
						MAX_PATH
					)

file_name = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
folder = '../images/kinetic_series/'

with AndoriXonUltra888(metaInformation) as andoriXonUltra888:
	stabalized_temp = andoriXonUltra888.cool_sensor()
	print("Temperature has stabalized at ", stabalized_temp)
	andoriXonUltra888.acquire_images()
	andoriXonUltra888.save_images(folder, file_name)
