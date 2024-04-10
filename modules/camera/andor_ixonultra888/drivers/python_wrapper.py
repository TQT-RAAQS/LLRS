from ctypes import *

# DLL_PATH = "./atmcd64d.dll"
DLL_PATH = "/usr/local/lib/libandor.so"

lib = CDLL(DLL_PATH)
print(lib)

loadlib = cdll.LoadLibrary(DLL_PATH)
path = c_char_p()
print(loadlib.Initialize(path))

k=input("press close to exit") 

class AndoriXonUltra888Driver:

	@staticmethod
	def initialize(
		dir
	):
		'''
		Initialize is used to initialize the Andor SDK System.
		The parameter dir is a char* pointer to the file path of the 
		DETECTOR.INI file. This parameter is only required on older cameras
		made by Andor and is not required on the iXonUltra888.
		'''
		try:
			dir = c_char_p(dir)
			lib.Initialize(dir)

		except Exception as e:
			raise e

	@staticmethod
	def get_camera_information(
		index,
		information
	):
		'''
		GetCameraInformation retireves if the camera is connected, if the dlls
		are loaded correctly and if the camera is initialized correctly.
		The first parameter is an integer representing the index of camera to 
		retrieve the information from.
		The second parameter is a long* pointer that will be returned with the 
		camera's state represented in 3 bits.
		'''
		try:
			index = c_int(index)
			information = c_long(information)

			lib.GetCameraInformation(
				index, 
				byref(information)
			)

			return information
		except Exception as e:
			raise e

	@staticmethod
	def get_temperature_range(
		mintemp,
		maxtemp
	):
		'''
		GetTemperatureRange returns the maximum and minimum temperature 
		cooling range of the camera. Both parameters are integer pointers
		and will be set with their respective values.
		'''
		try:
			mintemp = c_int(mintemp)
			maxtemp = c_int(maxtemp)

			lib.GetTemperatureRange(byref(mintemp), byref(maxtemp))

			return {
				"mintemp" : mintemp,
				"maxtemp" : maxtemp
			}
		except Exception as e:
			raise e

	@staticmethod
	def get_em_gain_range(
		low,
		high
	):
		'''
		GetEMGainRange Returns the minimum and maximum values of the current selected EM Gain mode and temperature of the sensor. 
		Both paramters are integers that are passed by reference representing the low and high values of the gain range.
		'''
		try:
			low = c_int(low)
			high = c_int(high)

			lib.GetEMGainRange(byref(low), byref(high))

			return {
				"low" : low,
				"high" : high
			}
		except Exception as e:
			raise e

	@staticmethod
	def get_number_ad_channels(
		channels
	):
		'''
		GetNumberADChannels returns the number of A-D converters available. Useful in HSSpeed retireval workflow.
		First parameter is an integer passed by reference representing the number of channels.
		'''
		try:
			channels = c_int(channels)

			lib.GetNumberADChannels(byref(channels))

			return channels
		except Exception as e:
			raise e

	@staticmethod
	def get_number_hs_speed(
		channel,
		typ,
		speeds
	):
		'''
		GetNumberHSSpeeds returns the number of horizontal shift speeds available. Channel is an integer, representing the AD channel. Typ is an integer, representing the type of output amplification (0 -> electron multiplication, 1 -> conventional). Speeds is an integer passed by reference, representing the number of allowed horizontal speeds.
		'''
		try:
			channel = c_int(channel)
			typ = c_int(typ)
			speeds = c_int(speeds)

			lib.GetNumberHSSpeeds(channel, typ, byref(speeds))

			return speeds
		except Exception as e:
			raise e

	@staticmethod
	def get_hs_speed(
		channel,
		typ,
		index,
		speed
	):
		'''
		GetHSSpeed returns the actual speeds available in MHz. Channel is an integer, representing the AD channel. Typ is an integer, representing the type of output amplification (0 -> electron multiplication, 1 -> conventional). Index is an integer representing which speed to retireve. Speed is a float passed by reference representing the horizontal shift speed in MHz.
		'''
		try:
			channel = c_int(channel)
			typ = c_int(typ)
			index = c_int(typ)
			speed = c_float(speed)

			lib.GetHSSpeed(channel, typ, index, byref(speed))

			return speed
		except Exception as e:
			raise e

	@staticmethod
	def get_number_vs_speeds(
        speeds
	):
		'''
        GetNumberVSSpeeds returns the number of vertical shift speeds available. Speeds is an integer passed by reference that represents the number of allowed vertical speeds.
		'''
		try:
			speeds = c_int(speeds)

			lib.GetNumberVSSpeeds(byref(speeds))

			return speeds
		except Exception as e:
			raise e

	@staticmethod
	def get_vs_speed(
		index,
		speed
	):
		'''
		GetVSSpeed returns the vertical shift speed in microseconds. Index is an integer representing which speed to return. Speed is a float passed by reference that represents the speed in microseconds per pixel shift.
		'''
		try:
			index = c_int(index)
			speed = c_float(speed)

			lib.GetVSSpeed(index, byref(speed))

			return speed
		except Exception as e:
			raise e
    
	@staticmethod
	def get_fastest_recommended_vs_speed(
    	index,
    	speed
	):
		'''
		GetFastestRecommendedVSSpeed returns the fastest speed which does not require the Vertical Clock Voltage to be adjusted. The values returned are the vertical shift speed index which is an integer passed by reference and the actual speed in microseconds per pixel shift which is a float passed by reference.
		'''
		try:
			index = c_int(index)
			speed = c_float(speed)

			lib.GetFastestRecommendedVSSpeed(byref(index), byref(speed))

			return {
				"index": index,
				"speed": speed
			}
		except Exception as e:
			raise e

	@staticmethod
	def cooler_on():
		'''
		Switch ON the cooling. Control is returned immediatly to the calling application
		'''
		try:
			lib.CoolerON()
		except Exception as e:
			raise e

	@staticmethod
	def set_temperature(
		temperature
	):
		'''
		SetTemperature will set the desired termperature of the detector. 
		'''
		try:
			temperature = c_int(temperature)
			
			lib.SetTemperature(temperature)
		except Exception as e:
			raise e

	@staticmethod
	def get_temperature(
		temperature
	):
		'''
		GetTemperature returns the temperature of the detector to the nearest degree. It also gives the status of the cooling process. Temperature is an integer passed by reference.
		'''
		try:
			temperature = c_int(temperature)

			lib.GetTemperature(byref(temperature))

			return temperature
		except Exception as e:
			raise e

	@staticmethod
	def set_cameralink_mode(
		mode
	):
		'''
		SetCameraLinkMode enables or disables the CameraLink output. Mode is an integer and 1 represents enabling and 0 represents disabling.
		'''
		try:
			mode = c_int(mode)

			lib.SetCameraLinkMode(mode)
		except Exception as e:
			raise e

	@staticmethod
	def set_trigger_mode(
		mode
	):
		'''
		SetTriggerMode will set the trigger mode that the camera will operate in. Mode is an integer representing the specific mode. 
		0 -> Integral
		1 -> External
		6 -> External Start
		7 -> External Exposure
		9 -> External FVB EM (only valid for EM Newton models in FVB mode) 
		10 -> Software Trigger
		'''
		try:
			mode = c_int(mode)

			lib.SetTriggerMode(mode)
		except Exception as e:
			raise e

	@staticmethod
	def set_em_gain_mode(
		mode
	):
		'''
		SetEMGainMode sets the EM Gain mode to one of the following settings. Mode is an integer and represents:
		0 -> The EM Gain is controlled by DAC settings in the range 0-255. Default mode.
 		1 -> The EM Gain is controlled by DAC settings in the range 0-4095.
 		2 -> Linear mode.
 		3 -> Real EM gain
 		'''
		try:
 		 	mode = c_int(mode)

			lib.SetEMGainMode(mode)
		except Exception as e:
 		 	raise e 

	@staticmethod
	def set_em_advanced(
 		state
	):
		'''
		SetEMAdvanced turns  on and off access to higher EM gain levels within the SDK. Typically, optimal signal to noise ratio and dynamic range is achieved between x1 to x300 EM Gain. Higher gains of > x300 are recommended for single photon counting only. Before using higher levels, you should ensure that light levels do not exceed the regime of tens of photons per pixel, otherwise accelerated ageing of the sensor can occur. 
		State is an integer, a value of 1 enables access and 0 disables access
		'''
		try:
			state = c_int(state)

			lib.SetEMAdvanced(state)
		except Exception as e:
			raise e

	@staticmethod
	def set_emccd_gain(
		gain
	):
		'''
		SetEMCCDGain allows the user to change the gain value given a range dependent on the gain mode. Gain is an integer representing the amount of gain applied.
		'''
		try:
			gain = c_int(gain)

			lib.SetEMCCDGain(gain)
		except Exception as e:
			raise e

	@staticmethod
	def set_baseline_clamp(
		state
	):
		'''
		SetBaselineClamp turns on and off the baseline clamp functionality. With this feature enabled the baseline level of each scan in a kinetic series will be more consistent across the sequence.
		State is an integer where 1 enables the baseline clamp and 0 disables the baseline clamp.
		'''
		try:
			state = c_int(state)

			lib.SetBaselineClamp(state)
		except Exception as e:
			raise e

	@staticmethod
	def set_frame_transfer_mode(
		mode
	):
		'''
		SetFrameTransferMode will set whether an acquisition will readout in Frame Transfer Mode. If the acquisition mode is Single Scan or Fast Kinetics this call will have no affect. 
		Mode is an integer where 1 turns ON and 0 turns OFF.
		'''
		try:
			mode = c_int(mode)

			lib.SetFrameTransferMode(mode)
		except Exception as e:
			raise e

	@staticmethod
	def set_hs_speed(
		typ,
		index
	):
		'''
		SetHSSpeed will set the speed at which the pixels are shifted into the output node during the readout phase of an acquisition.
		Typ is an integer that represents the type of output amplification where 0 is electron multiplication and 1 is conventional. 
		Index is an integer that represents the horizontal speed to be used from 0 to GetNumberHSSpeeds()-1.
		'''
		try:
			typ = c_int(typ)
			index = c_int(index)

			lib.SetHSSpeed(typ, index)
		except Exception as e:
			raise e

	@staticmethod
	def set_vs_speed(
		index
	):
		'''
		SetVSSpeed will set the vertical shift speed to be used.
		Index is an integer that represents which vertical speed to be used from 0 to GetNumberVSSpeeds()-1.
		'''
		try:
		 	index = c_int(index)

		 	lib.SetVSSpeed(index)
		except Exception as e:
		 	raise e 

	@staticmethod
	def set_acquisition_mode(
 		mode
	):
		'''
		SetAcquisitionMode sets the acquisition mode to be used on the next acquisition. Mode is an integer with valid values:
		1 -> Single Scan
		2 -> Accumuluate
		3 -> Kinetics
		4 -> Fast Kinetics
		5 -> Run till abort
		'''
		try:
			mode = c_int(mode)

			lib.SetAcquisitionMode(mode)
		except Exception as e:
			raise e

	@staticmethod
	def set_read_mode(
		mode
	):
		'''
        SetReadMode sets the readout mode to be used on the subsequent acquisitions. Mode is an integer and represents values:
        0 -> Full Vertical Binning
        1 -> Multi-Track
        2 -> Random-Track
        3 -> Single-Track
        4 -> Image
        '''
		try:
			mode = c_int(mode)

			lib.SetReadMode(mode)
		except Exception as e:
			raise e

	@staticmethod
	def set_image_params(
    	hbin,
    	vbin,
    	hstart,
    	hend,
    	vstart,
    	vend
	):
		'''
		SetImage will set the horizontal and vertical binning to be used when taking a full resolution image. All paramters are integers.
		hbin represents the number of pixels to bin horizontally
		vbin represents the number of pixels to bin vertically
		hstart represents the start column (inclusive)
		hend represents the end column (inclusive)
		vstart represents the start row (inclusive)
		vend represents the end row (inclusive)
		'''
		try:
			hbin = c_int(hbin)
			vbin = c_int(vbin)
			hstart = c_int(hstart)
			hend = c_int(hend)
			vstart = c_int(vstart)
			vend = c_int(vend)

			lib.SetImage(hbin, vbin, hstart, hend, vstart, vend)
		except Exception as e:
			raise e

	@staticmethod
	def set_shutter(
		typ,
		mode,
		closingtime,
		openingtime
	):
		'''
		SetShutter controls the behaviour of the shutter.
		
	@staticmethod
	def get_status():
		try:
			lib.GetStatus()
		except:
			print("Error")
			raise Exception()
			'''