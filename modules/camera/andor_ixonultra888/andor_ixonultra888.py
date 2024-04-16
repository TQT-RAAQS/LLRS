"""
Performs all utility from the Andor SDK.
Future: can look into using events to replace while loop busy waiting

This class should primarily be inputted into ImageAcquisition

Sailesh Bechar
Fall 2020

Brooke Dolny
Winter 2021

Jonathon Kambulow
Fall 2021
"""
import datetime
import sys
from multiprocessing.pool import ThreadPool

import numpy as np
from experiment.instruments.camera import Camera
from experiment.instruments.camera.andor_ixonultra888.emccd_meta_information import \
    EmccdMetaInformation
from experiment.instruments.camera.andor_ixonultra888.image import Image
from experiment.instruments.camera.andor_ixonultra888.maker_files.software.pyAndorSDK2.atmcd import \
    atmcd
import time


class AndorException(Exception):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class AndoriXonUltra888(Camera):
    def __init__(self):
        """
        Opens the SDK handle and stores the board's meta information
        """
        self.meta_information = {}  # this must be stored because therea aren't many getters
        self.sdk = atmcd()  # load the atmcd library
        super(AndoriXonUltra888, self).__init__()

    def __enter__(self):

        # initialize camera
        if sys.platform == 'linux':
            (ret) = self.sdk.Initialize('/usr/local/etc/andor')
        elif sys.platform == 'win32':
            (ret) = self.sdk.Initialize('')

        if self.sdk.DRV_SUCCESS != ret:
            self.initialized = False
            raise AndorException(ret, "Failed to Initialize Andor iXon SDK")

        self.initialized = True
        self.update_pixel_size()
        # default to 50 until set_acquisition_properties is called
        self.image_buffer = np.zeros(
            (50, self.xpixels, self.ypixels), dtype=np.dtype(np.int16),)
        self.curr_image = 0
        return self

    def __exit__(self, type, value, traceback):
        self.sdk.SetShutter(1, 2, 0, 0)  # always close shutter
        self.shutdown_camera()
        return type is None

    def open_connection(self):
        self.__enter__()

    def close_connection(self):
        self.__exit__(None, None, None)

    def update_pixel_size(self):
        (ret, self.xpixels, self.ypixels) = self.sdk.GetDetector()
        return ret
    
    def set_roi(self, roi, binning):
        xpixels = roi["width"]
        ypixels = roi["height"]
        xoffset = roi.get("left")
        yoffset = roi.get("bottom")

        if binning is None:
            hbin, vbin = 1, 1
        else:
            hbin = binning.get("hbin", 1)
            vbin = binning.get("vbin", 1)

        if xoffset is None:
            xoffset = (1024 - xpixels) // 2 + 1
        if yoffset is None:
            yoffset = (1024 - ypixels) // 2 + 1
        self.sdk.SetIsolatedCropModeType(1)
        ret = self.sdk.SetIsolatedCropModeEx(
            1, 
            ypixels, 
            xpixels, 
            vbin, 
            hbin,
            xoffset,
            yoffset
        )
        self.update_pixel_size()
        return ret

    def shutter_manual_control(self,
                               type: int, 
                               mod: int, 
                               closing_time: float = 27,
                               openning_time: float = 27,
                               extmode: int = None):
        """
          typ - shutter type:
            1 - Output TTL high signal to open shutter
            0 - Output TTL low signal to open shutter
          mode - Shutter mode:
            0 - Automatic
            1 - Open
            2 - Close
          closingtime - Time shutter takes to close (milliseconds)
          openingtime - Time shutter takes to open (milliseconds)
          extmode - Shutter mode:
            None - External shutter not configured
            0 - Automatic
            1 - Open
            2 - Close
        """
        print(f"Shutter mode: {type}, {mod}, {closing_time}, {openning_time}, {extmode}")
        if extmode is None:
            if mod == 0:
                # For this camera, closing and opening times MUST be 27 ms when type is automatic.
                closing_time = 27
                openning_time = 27
            return self.sdk.SetShutter(
                typ = type,
                mode = mod,
                closingtime=closing_time,
                openingtime=openning_time
            )
        else:
            return self.sdk.SetShutterEx(
                type,
                mod,
                closingtime=closing_time,
                openingtime=openning_time,
                extmode=extmode
            )

    def get_camera_information(self):
        return self.sdk.GetCapabilities()

    def is_trigger_mode_available(self, mode):
        ret = self.sdk.IsTriggerModeAvailable(mode)
        return ret == self.sdk.DRV_SUCCESS

    def cool_sensor(self, blockUntilStabilized: bool = True):
        """
        Cools sensor to temperature provided in meta information. Blocks until stabalized

        Returns
        -------
        tempf (int) : Most recent temperature of sensor
        """
        (ret, min_temp, max_temp) = self.sdk.GetTemperatureRange()
        print(
            "Camera temp range returned ", ret, " with low ", min_temp, " and max ", max_temp,
        )

        (ret) = self.sdk.SetTemperature(self.meta_information['temperature'])
        print(
            "Camera temperature set to", self.meta_information['temperature'], "with status ", ret,
        )

        (ret) = self.sdk.CoolerON()
        print("Cooler on with status ", ret)

        (ret, temp_f) = self.sdk.GetTemperatureF()
        print("Camera temp returned ", ret, " with value ", temp_f)

        if blockUntilStabilized:
            old_temp = temp_f
            counter = 0
            while ret != self.sdk.DRV_TEMP_STABILIZED:
                (ret, temp_f) = self.sdk.GetTemperatureF()
                counter = (counter + 1) % 2000
                if counter == 0:
                    print("Got a temperature of: " + str(temp_f) + ", with code: " + str(ret))
                if old_temp - temp_f > 5:
                    old_temp = temp_f
                    print(temp_f)

        return int(temp_f)

    def set_static_properties(self, config_dict):
        """
        Sets properties that should not vary from acquisition to acquisiton
        """
        self.meta_information.update(config_dict)
        self.meta_information['pixel depth'] = 16

        if config_dict.get('temperature', None) is not None:
            (ret) = self.sdk.SetTemperature(
                self.meta_information['temperature'])
            print('SetTemperature with status', ret)
            self.cool_sensor(self.meta_information.get('block_until_temperature_stabilized', True))

        if config_dict.get('em_gain_mode') is not None:
            (ret) = self.sdk.SetEMGainMode(
                self.meta_information['em_gain_mode'])
            print("SetEMGainMode on with status ", ret)
            status = ret and self.sdk.DRV_SUCCESS

        if config_dict.get('em_advanced') is not None:
            (ret) = self.sdk.SetEMAdvanced(
                self.meta_information['em_advanced'])
            print("SetEMAdvanced with status", ret)
            status = ret and status

        (ret, gain_min, gain_max) = self.sdk.GetEMGainRange()
        print(
            "Camera gain range returned ", ret, " with low ", gain_min, " and max ", gain_max,
        )
        # status = ret and status

        if config_dict.get('em_gain') is not None:
            (ret) = self.sdk.SetEMCCDGain(self.meta_information['em_gain'])
            print("EMCCD gain set ",
                  self.meta_information['em_gain'], "with status", ret)
            status = ret and status

        if config_dict.get('trigger_mode') is not None:
            (ret) = self.sdk.SetTriggerMode(
                self.meta_information['trigger_mode'])
            print(
                "Function SetTriggerMode returned", ret, "mode =",
                self.get_trigger_mode(self.meta_information['trigger_mode'])
            )
            status = ret and status

        if config_dict.get('frametransfer_mode') is not None:
            (ret) = self.sdk.SetFrameTransferMode(
                self.meta_information['frametransfer_mode'])
            print("Function SetFrameTransferMode returned", ret)
            status = ret and status

        if config_dict.get('baseline_clamp') is not None:
            (ret) = self.sdk.SetBaselineClamp(
                self.meta_information['baseline_clamp'])
            print("Function SetBaselineClamp returned", ret)
            status = ret and status

        if config_dict.get('cameralink_mode') is not None:
            (ret) = self.sdk.SetCameraLinkMode(
                self.meta_information['cameralink_mode'])
            print("Function SetCameraLinkMode returned", ret)
            status = ret and status

        if config_dict.get('cooler_mode') is not None:
            (ret) = self.sdk.SetCoolerMode(
                self.meta_information['cooler_mode'])
            print("Function SetCoolerMode returned", ret)
            status = ret and status
        
        if config_dict.get('fan_mode') is not None:
            (ret) = self.sdk.SetFanMode(
                self.meta_information['fan_mode'])
            print("Function SetFanMode returned", ret)
            status = ret and status
        
        print('Attempting to change settings', config_dict)

        return ret

    def get_status(self):
        """
        Get the current status of the AndoriXonUltra888 camera.
        """
        (ret, status) = self.sdk.GetStatus()
        return status

    def set_acquisition_properties(self, config_dict):
        """
        Sets properties that may vary if planning on multiple acqusitions
        """
        self.meta_information.update(config_dict)

        # specify meta information based on configuration data
        VS_SPEED_VALS = ["0.6us", "1.12us", "2.2us", "4.3us"]
        if self.meta_information['amp_index'] == 0:
            SENSITIVITY_VALS = [18.2, 5.54, 16.4, 4.47, 16.3, 4.00, 16.1, 3.89]
            HS_SPEED_VALS = ["30MHz", "20MHz", "10MHz", "1MHz"]
        else:
            SENSITIVITY_VALS = [3.33, 0.8, 3.34, 0.8]
            HS_SPEED_VALS = ["1MHz", "0.1MHz"]
        self.meta_information['sensitivity'] = SENSITIVITY_VALS[
            2 * self.meta_information['hs_speed_index'] +
            self.meta_information['preamp_gain_index']
        ]
        self.meta_information['hs_speed'] = HS_SPEED_VALS[self.meta_information['hs_speed_index']]
        self.meta_information['vs_speed'] = VS_SPEED_VALS[self.meta_information['vs_speed_index']]

        # Verify camera is not still acquiring
        status = 0
        while status != atmcd.DRV_IDLE:
            (ret, status) = self.sdk.GetStatus()

        if self.meta_information.get('acquisition_mode') is not None:
            # set acquisition parameters on camera
            (ret) = self.sdk.SetAcquisitionMode(
                self.meta_information['acquisition_mode'])
            print("Function SetAcquisitionMode returned", ret)

        if self.meta_information.get('readout_mode') is not None:
            (ret) = self.sdk.SetReadMode(self.meta_information['readout_mode'])
            print("Function SetReadMode returned", ret, "mode = Image")

        binning = self.meta_information.get("binning", {})
        hbin, vbin = binning.get("hbin", 1), binning.get("vbin", 1)
        (ret) = self.sdk.SetImage(hbin, vbin, 1, self.xpixels, 1, self.ypixels)
        print(
            "Function SetImage returned",
            ret,
            "hbin = 1 vbin = 1 hstart = 1 hend =",
            self.xpixels,
            "vstart = 1 vend =",
            self.ypixels,
        )

        if self.meta_information.get('shutter_mode') is not None:
            shutter_mode = self.meta_information.get('shutter_mode')
            inf = self.meta_information.get("shutter_external")
            if inf and inf != -1:
                ret = self.shutter_manual_control(
                    inf["ttl"],
                    shutter_mode,
                    inf["closing_time"],
                    inf["openning_time"],
                    inf["shutter_mode"]
                )
            else:
                ret = self.shutter_manual_control(1, shutter_mode)
            print(f"Function SetShutter returned {ret}, mode = {self.meta_information['shutter_mode']}")
            print(f"External function mode: {inf}")

        if self.meta_information.get('exposure_time') is not None:
            (ret) = self.sdk.SetExposureTime(
                self.meta_information['exposure_time'])
            print(
                "Function SetExposureTime returned", ret, "time =", self.meta_information[
                    'exposure_time'],
            )

        if self.meta_information.get('amp_index') is not None:
            (ret) = self.sdk.SetOutputAmplifier(
                self.meta_information['amp_index'])
            print(
                "Function SetOutputAmplifier returned", ret, "with index", self.meta_information[
                    'amp_index'],
            )

        if self.meta_information.get('preamp_gain_index') is not None:
            (ret) = self.sdk.SetPreAmpGain(
                self.meta_information['preamp_gain_index'])
            print(
                "Function SetPreAmpGain returned", ret, "with index", self.meta_information[
                    'preamp_gain_index'],
            )

        # TODO why does this extra condition exist??
        if self.meta_information.get('hs_speed_index') is not None and self.meta_information.get('amp_index'):
            (ret) = self.sdk.SetHSSpeed(
                self.meta_information['amp_index'], self.meta_information['hs_speed_index'])
            print(
                "Function SetHSSpeed returned", ret, "with index", self.meta_information[
                    'hs_speed_index'],
            )

        if self.meta_information.get('vs_speed_index') is not None:
            (ret) = self.sdk.SetVSSpeed(self.meta_information['vs_speed_index'])
            print(
                "Function SetVSSpeed returned", ret, "with index", self.meta_information['vs_speed_index'],
            )
            
        if self.meta_information.get('num_kinetics') is not None:
            (ret) = self.sdk.SetNumberKinetics(self.meta_information['num_kinetics'])
            print(
                "Function SetNumberKinetics returned", ret, "with index", self.meta_information['num_kinetics'],
            )

        if self.meta_information.get("roi"):
            ret = self.set_roi(self.meta_information.get("roi"), self.meta_information.get("binning"))
            print(ret, "Changing the ROI to: ", self.meta_information.get("roi"))

        (ret) = self.sdk.PrepareAcquisition()
        print("Function PrepareAcquisition returned", ret)
        
        self.update_pixel_size()

    def setup(self, config_dict):
        self.meta_information.update(config_dict)

    def set_image_count(self, image_count, image_delay):
        """
        Sets the number of images to take and the time between images, and
        creates the image_buffer array
        """
        self.meta_information['num_accumulations'] = image_count
        self.meta_information['kinetic_cycle_time'] = image_delay
        self.meta_information['num_kinetics'] = image_count

        (ret) = self.sdk.SetAccumulationCycleTime(
            self.meta_information['kinetic_cycle_time'])
        print("Function SetAccumulationCycletime returned", ret)

        (ret) = self.sdk.SetNumberAccumulations(
            self.meta_information['num_accumulations'])
        print("Function SetNumberAccumulations returned", ret)

        (ret) = self.sdk.SetNumberKinetics(
            self.meta_information['num_kinetics'] + 1)
        print("Function SetNumberKinetics returned", ret)

        (ret) = self.sdk.SetKineticCycleTime(
            self.meta_information['kinetic_cycle_time'])
        print("Function SetKineticCycleTime returned", ret)

        self.image_buffer = np.zeros(
            (self.xpixels, self.ypixels,
             self.meta_information['num_kinetics']), dtype=np.dtype(np.int16),
        )

    def acquire_single_image(self):
        '''
        Acquire a single image

        Returns:
        --------
        result: ndarray
            Contains all collected image data
        metadata: dict
            dictionary of metadata
        now: datetime
            time of the start of the acquisition
        '''
        if self.meta_information['acquisition_mode'] == 5:
            now = self._acquire_images_rta()
        elif self.meta_information['acquisition_mode'] == 1:
            now = self._acquire_images_ss()
        else:
            self.set_image_count(1, 1)
            now = self._acquire_images()
        return self.image_buffer, self.meta_information, now

    def acquire_images(self, image_count, image_delay):
        '''
        Acquire a number of images with a specific delay between each.

        Parameters:
        ----------
        image_count: int
            number of images to take
        image_delay: float
            time in seconds between in picture

        Returns:
        --------
        result: ndarray
            Contains all collected image data
        metadata: dict
            dictionary of metadata
        now: datetime
            time of the start of the acquisition
        '''
        self.set_image_count(image_count, image_delay)

        # stabalized_temp = self.cool_sensor() #TODO confirm if this can be removed
        # print("Temperature has stabalized at ", stabalized_temp)
        now = self._acquire_images()
        return self.image_buffer, self.meta_information, now

    def listen_for_hardware_trigger(self, image_count=0, line=0, delay: int = 0):
        # Parameters:
            # Line: Meaningless for EMCCD

        # pool = ThreadPool(processes=1)
        # self.async_result = pool.apply_async(self.acquire_images, (image_count, delay)) #TODO allow for more delay options, possibly enter it as a variable

        if self.meta_information['trigger_mode'] == 0:
            raise Exception('Please set the trigger mode to external trigger')
        
        if self.meta_information['acquisition_mode'] == 3:
            self.sdk.SetNumberKinetics(image_count)
            self.start_acquisition()
        elif self.meta_information['acquisition_mode'] == 5:
            self.start_acquisition()
        else:
            raise Exception('Please set the acquisition mode to kinetic mode')
        
        return None, None, datetime.datetime.now()

    def stop_listening_for_hardware_trigger(self, timeout=None):
        # Timeout not implemented.
        # now = self.async_result.get()
        # return self.image_buffer, self.meta_information, now
        self.abort_acquisition()
        return self.get_all_acquired_images()

    def _acquire_images(self, trigger_mode: str = 'internal'):
        '''
        Starts acquisition given preset parameters. Stores images to self.image_buffer

        Parameters
        ----------
        trigger_mode: str
            'internal' or 'external'
        '''
        image_size = self.xpixels * self.ypixels

        # vt = win32event.CreateEvent(None,0,0,None) # Handle if wanting to use events to pass into sdk.SetDriverEvent()

        # Verify camera is not still acquiring
        status = 0
        while status != atmcd.DRV_IDLE:
            (ret, status) = self.sdk.GetStatus()

        now = datetime.datetime.now()
        if trigger_mode == 'internal':
            (ret) = self.sdk.StartAcquisition()
            print("Function StartAcquisition returned", ret)

        self.curr_image = 0

        while self.curr_image < self.meta_information['num_kinetics']:
            (ret) = self.sdk.WaitForAcquisition()
            print("Function WaitForAcquisition returned", ret)

            (ret, first, last) = self.sdk.GetNumberNewImages()
            print(
                "Function GetNumberNewImages returned", ret, "first =", first, "last =", last,
            )
            if first != last:
                print(
                    "ERROR! Kinetic Cycle Time too fast", self.meta_information['kinetic_cycle_time'],
                )
                break
            (ret, full_frame_buffer, validfirst,
             validlast) = self.sdk.GetImages(last, last, image_size)
            print(
                "Function GetImages returned",
                ret,
                "first pixel =",
                full_frame_buffer[0],
                "size =",
                image_size,
                validfirst,
                validlast,
            )

            self.image_buffer[:, :, last - 1] = np.array(full_frame_buffer, dtype=np.dtype(np.int16)).reshape(
                self.xpixels, self.ypixels
            )
            print(np.max(self.image_buffer))
            self.curr_image = last
        return now

    def _acquire_images_rta(self, timeout: bool = True):
        '''
        Runs image acquisition in Run Till Abort acquisition mode.
        This command must be run AFTER listen_for_hardware_trigger()

        Parameters
        ----------
        timeout: bool
            Recommended to keep it True
        '''
        self.image_buffer = np.zeros(
            (self.xpixels, self.ypixels), dtype=np.int16)
        if timeout:
            ret_timeout = self.wait_for_acquisition_timeout(10000)
            now = datetime.datetime.now()
        else:
            ret_timeout = self.wait_for_acquisition()
            now = datetime.datetime.now()
        if ret_timeout != 20002:
            self.stop_listening_for_hardware_trigger()
            print('Aborting acquisition')
            raise Exception(
                'Nothing was acquired within timeout of', timeout, 's')
        (ret, first, last) = self.sdk.GetNumberNewImages()
        image_size = self.xpixels * self.ypixels
        (ret, full_frame_buffer, validfirst,
         validlast) = self.sdk.GetImages16(last, last, image_size)
        self.image_buffer = np.array(full_frame_buffer, dtype=np.int16).reshape(
            self.xpixels, self.ypixels
        )
        self.curr_image = last
        return now

    def _acquire_images_ss(self):
        '''
        Runs image acquisition when the camera is in single scan mode
        The exposure time is set by software and the camera completes acquisition after one scan
        '''
        now = datetime.datetime.now()
        self.start_acquisition()  # triggers acquisition on camera
        ret = self.wait_for_acquisition()
        image_size = self.xpixels * self.ypixels
        (ret, full_frame_buffer, validfirst,
         validlast) = self.sdk.GetImages(1, 1, image_size)
        self.image_buffer = np.array(full_frame_buffer, dtype=np.int16).reshape(
            self.xpixels, self.ypixels
        )
        return now

    def start_acquisition(self):
        start = time.time()
        ret = self.sdk.StartAcquisition()
        #print("time to start acq: " + str(time.time() - start))
        #print('start_acquisition returned', ret)
        return ret

    def wait_for_acquisition(self):
        ret = self.sdk.WaitForAcquisition()
        return ret

    def wait_for_acquisition_timeout(self, time):
        '''
        Parameters
        ----------
        time: float
            Time to wait until timeout (in ms)
        '''
        ret = self.sdk.WaitForAcquisitionTimeOut(time)
        print('wait_for_acquisition_timeout returned', ret)
        return ret

    def retrieve_all_data(self):
        '''Returns number of new images acquired'''
        (ret, first, nr_images) = self.sdk.GetNumberNewImages()
        image_buffer_size = self.xpixels * self.ypixels * nr_images
        (ret, full_frame_buffer, validfirst, validlast) = self.sdk.GetImages(
            first, nr_images, image_buffer_size)
        self.image_buffer = np.array(full_frame_buffer, dtype=np.int16).reshape(
            self.xpixels, self.ypixels, nr_images
        )
        now = datetime.datetime.now()
        return now

    def save_image_data(self, folder, filename, filetype='png'):
        """
        Pickles the image buffer and meta information to file

        Parameters
        -----------
        folder : str
            destination path where the file will be saved
        filename:  str
            name of the file to be saved, without extension

        """
        kinetic_series = Image(self.meta_information, self.image_buffer)
        if filetype == 'png':
            kinetic_series.save_all_images(folder, filename)
        else:
            kinetic_series.save_pickle(folder, filename)

    def save_most_recent_image(self, folder, filename):
        """
        Saves the most recent image as a png

        Args
        -----------
        folder(str) : destination path where the file will be saved
        filename(str) : name of the file to be saved, without extension
        """
        kinetic_series = Image(self.meta_information, self.image_buffer)
        kinetic_series.save_most_recent_image(folder, filename)

    def display_most_recent_image(self):
        """
        Displays the most recent image
        """
        kinetic_series = Image(self.meta_information, self.image_buffer)
        kinetic_series.display_most_recent_image()

    def shutdown_camera(self):
        """
        Safely turns off SDK and cooler
        """
        (ret, temp_f) = self.sdk.GetTemperatureF()
        print("Camera temp returned ", ret, " with value ", temp_f)

        (ret) = self.sdk.ShutDown()
        print("Shutdown returned", ret)

    def cooler_off(self):
        '''
        Turns cooler off
        '''
        (ret) = self.sdk.CoolerOFF()
        print("Cooler off with status ", ret)
        self.get_temperature()

    def get_temperature(self):
        '''
        Retrieves current temperature of the sensor
        '''
        (ret, temp_f) = self.sdk.GetTemperatureF()
        # print('Camera temp returned ', ret, ' with value ', temp_f)
        return ret, temp_f

    def abort_acquisition(self):
        """
        Aborts current acquisition and returns index to retrieve image buffer

        Returns

        -----------
        curr_image(int) : index of most recent image acquired
        """
        (ret) = self.sdk.AbortAcquisition()
        print("AbortAcquisition returned", ret)
        return self.curr_image  # Return index to retrieve images acquired in image buffer
    
    def get_all_acquired_images(self):
        _, number = self.sdk.GetTotalNumberImagesAcquired()
        now = datetime.datetime.now()

        if number == 0:
            return None, self.meta_information, now
        else:
            _, images, _, _ = self.sdk.GetImages(1, number, number * self.xpixels * self.ypixels)
            images = np.array(np.reshape(images, (self.xpixels, self.ypixels, number), "F"), dtype="uint16")

            return images, self.meta_information, now

    def get_most_recent_image(self):
        now = datetime.datetime.now()
        
        image = self.sdk.GetMostRecentImage(self.xpixels * self.ypixels)[1]
        image = np.array(np.reshape(np.array(image), (self.xpixels, self.ypixels)), dtype="uint16")
        if np.any(image):
            return image, self.meta_information, now
        else:
            return None, self.meta_information, now

    def get_image_buffer(self):
        """
        Returns contents of image buffer

        Returns

        -----------
        image_buffer(np array) : dimensions (xpixels, ypixels, num images)
        """
        return self.image_buffer

    def get_fastest_vs_speed(self):
        """
        Retrieves recommended fastest vs speed from the SDK

        Returns

        -----------
        ret (int) : status of operation (20002 = success)
        fastest_vs_speed_index (int) : vs_speed index
        speed (int) : speed in us per pixel shift
        """
        (ret, fastest_vs_speed_index, speed) = self.sdk.GetFastestRecommendedVSSpeed()
        print(
            "Function GetFastestRecommendedVSSpeed returned", ret, "Index", fastest_vs_speed_index, "speed", speed,
        )
        return ret, fastest_vs_speed_index, speed

    def get_all_hs_speeds(self):
        """
        Retrieves all horizontal shift speeds from the SDK

        Returns

        -----------
        hs_speed_list(list) : speeds in MHz
        """
        (ret, num_AD_channels) = self.sdk.GetNumberADChannels()
        print("Function GetNumberADChannels returned",
              ret, "with number", num_AD_channels)
        (ret, num_amp) = self.sdk.GetNumberAmp()
        print("Function GetNumberAmp returned", ret, "with number", num_amp)
        (ret, num_preamp_gains) = self.sdk.GetNumberPreAmpGains()
        print(
            "Function GetNumberPreAmpGains returned", ret, "with number", num_preamp_gains,
        )

        hs_speed_list = []
        for i in range(num_AD_channels):
            (ret, num_hs_speeds) = self.sdk.GetNumberHSSpeeds(i, 0)
            print(
                "Function GetNumberHSSpeeds returned", ret, "with number of speeds", num_hs_speeds,
            )
            for j in range(num_amp):
                for k in range(num_hs_speeds):
                    (ret, hs_speed) = self.sdk.GetHSSpeed(i, j, k)
                    if ret == atmcd.DRV_SUCCESS:  # Sometimes does not produce valid values
                        hs_speed_list.append(hs_speed)
                        print(
                            "Function GetHSSpeed returned", ret, "with speed", hs_speed, "MHz",
                        )
                    for m in range(num_preamp_gains):
                        (ret, preamp_gain) = self.sdk.GetPreAmpGain(m)
                        if ret == atmcd.DRV_SUCCESS:
                            print(
                                "Function GetPreAmpGain returned", ret, "with gain", preamp_gain,
                            )
        return hs_speed_list

    def get_all_vs_speeds(self):
        """
        Retrieves all vertical shift speeds from the SDK

        Returns

        -----------
        vs_speed_list(list) : speeds in us per pixel shift
        """
        (ret, num_vs_speeds) = self.sdk.GetNumberVSSpeeds()
        print("Function GetNumberVSSpeeds returned",
              ret, "with number", num_vs_speeds)

        vs_speed_list = []
        for i in range(num_vs_speeds):
            (ret, vsspeed) = self.sdk.GetVSSpeed(i)
            print(
                "Function GetVSSpeed returned", ret, "with speed", vsspeed, " microseconds per pixel shift",
            )
            vs_speed_list.append(vsspeed)
        return vs_speed_list

    def get_minimum_achievable_temperature(self):
        """
        Set temperature to -120 C and measure the lowest temperature on sensor after 10 mins

        Returns
        ---------
        temp (int) : highest lowest temperature for all settings
        """
        # Preamp gain 0 Amp index 0 Hs index 0 temp = -85
        # Preamp gain 0 Amp index 0 Hs index 1 temp = -91
        # Preamp gain 0 Amp index 0 Hs index 2 temp = -98
        # Preamp gain 0 Amp index 0 Hs index 3 temp = -103
        return -85

    @staticmethod
    def get_trigger_mode(index):
        '''
        Helper function to return trigger mode names based on index input
        '''
        if index == 0:
            return 'Internal'
        elif index == 1:
            return 'External'
        elif index == 6:
            return 'External Start'
        elif index == 7:
            return 'External Exposure'
        elif index == 10:
            return 'Software Trigger'
