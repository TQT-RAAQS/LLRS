from experiment.instruments.camera.andor_ixonultra888.maker_files.software.pyAndorSDK2.atmcd import AndorCapabilities
from experiment.instruments.camera.andor_ixonultra888.andor_ixonultra888 import AndoriXonUltra888
from experiment.instruments.camera.andor_ixonultra888.emccd_meta_information import EmccdMetaInformation

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

def check_camera_properties(camera):
	ret, caps = camera.get_camera_information()
	print("ulSize:\t  "+bin(caps.ulSize))
	print("ulSize:\t  ", bin(caps.ulSize))
	print("ulAcqModes:\t  ", bin(caps.ulAcqModes))
	print("ulReadModes:\t  ", bin(caps.ulReadModes))
	print("ulTriggerModes:\t  ", bin(caps.ulTriggerModes))
	print("ulCameraType:\t  ", bin(caps.ulCameraType))
	print("ulPixelMode:\t  ", bin(caps.ulPixelMode))
	print("ulSetFunctions:\t  ", bin(caps.ulSetFunctions))
	print("ulGetFunctions:\t  ", bin(caps.ulGetFunctions))
	print("ulFeatures:\t  ", bin(caps.ulFeatures))
	print("ulPCICard:\t  ", bin(caps.ulPCICard))
	print("ulEMGainCapability:\t  ", bin(caps.ulEMGainCapability))
	print("ulFTReadModes:\t  ", bin(caps.ulFTReadModes))
	print("ulFeatures2:\t  ", bin(caps.ulFeatures2))
	print("Trigger Mode:\t ", str(camera.get_trigger_mode(10)))

if __name__ == "__main__":
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
	with AndoriXonUltra888(metaInformation) as camera:
		check_camera_properties(camera)
