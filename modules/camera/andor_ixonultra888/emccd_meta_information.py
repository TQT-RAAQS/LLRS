# this class is no longer being used because:
# - it makes saving and loading to and from yaml files difficult
# - it doesn't add much more functionality over a dictionary
# - saving this to a file requires pickling it. The pickled information isn't easily viewable
class EmccdMetaInformation:
    """
    Meta Information to be stored when acquiring images on EMCCD camera

    Sailesh Bechar
    Fall 2020
    """

    def __init__(
        self,
        exposure_time=None,
        preamp_gain_index=None,
        hs_speed_index=None,
        vs_speed_index=None,
        em_gain=None,
        em_advanced=None,
        em_gain_mode=None,
        acquisition_mode=None,
        num_accumulations=None,
        readout_mode=None,
        num_kinetics=None,
        kinetic_cycle_time=None,
        temperature=None,
        trigger_mode=None,
        amp_index=None,
        baseline_clamp=None,
        cameralink_mode=None,
        frametransfer_mode=None,
        max_path=None,
    ):
        """
        Initializes EMCCD Meta Information class and sets relavant member variables
        For complete description of arguments see Andor SDK Manual

        Args

        ----------
        exposure_time (float) : time in ms (0.0001 seconds is too short for actual acquisition)
        preamp_gain_index (int) : 0 - Gain 1.0, 1 - Gain 2.0
        hs_speed_index (int) : values for index found in getHSSpeed() member function
        vs_speed_index (int) : values for index found in getVSSpeed() member function
        em_gain (int) : em gain of camera
        em_advanced (int) : 1 - Enable access to higher levels of EM gain (above 300)
        em_gain_mode (int) : 0 - The EM Gain is controlled by DAC settings in the range 0-255. Default mode
                                1 - The EM Gain is in the range 0-4095, 2 - Linear mode, 3 - Real EM gain
        acquisition_mode (int) : 3 - Kinetic Series
        num_accumulations (int) : how many images to accumalate together
        readout_mode (int) : 4 - Image
        num_kinetics (int) : number of images to take in kinetic series
        kinetic_cycle_time (float) : time in ms per each kinetic series image
        temperature (int) : temperature to set cooler
        trigger_mode (int) : 0 - Internal
        amp_index (int) : 0 - EM, 1 - Conventional
        baseline_clamp (int) : 1 enables
        cameralink_mode (int) : 1 enables
        frametransfer_mode (int) : 1 enables
        max_path (string) : maximum length of string the SDK will write to
        """
        self.exposure_time = exposure_time
        self.preamp_gain_index = preamp_gain_index
        self.hs_speed_index = hs_speed_index
        self.vs_speed_index = vs_speed_index
        self.em_gain = em_gain
        self.em_advanced = em_advanced
        self.em_gain_mode = em_gain_mode
        self.acquisition_mode = acquisition_mode
        self.num_accumulations = num_accumulations
        self.readout_mode = readout_mode
        self.num_kinetics = num_kinetics
        self.kinetic_cycle_time = kinetic_cycle_time
        self.temperature = temperature
        self.trigger_mode = trigger_mode
        self.amp_index = amp_index
        self.baseline_clamp = baseline_clamp
        self.cameralink_mode = cameralink_mode
        self.frametransfer_mode = frametransfer_mode
        self.max_path = max_path
        self.sensitivity = self.get_sensitivity()

    def verify(self):
        return (
            self.exposure_time is not None
            and self.preamp_gain_index is not None
            and self.hs_speed_index is not None
            and self.vs_speed_index is not None
            and self.em_gain is not None
            and self.em_advanced is not None
            and self.em_gain_mode is not None
            and self.acquisition_mode is not None
            and self.num_accumulations is not None
            and self.readout_mode is not None
            and self.num_kinetics is not None
            and self.kinetic_cycle_time is not None
            and self.temperature is not None
            and self.trigger_mode is not None
            and self.amp_index is not None
            and self.baseline_clamp is not None
            and self.cameralink_mode is not None
            and self.frametransfer_mode is not None
            and self.max_path is not None
            and self.sensitivity is not None
        )

    def configure(self, config_dict):
        for k, val in config_dict.items():
            if hasattr(self, k) and not k.startswith("__") and not callable(k):
                setattr(self, k, val)
        self.sensitivity = self.get_sensitivity()

    def get_hs_speed(self):
        if self.amp_index == 0:
            HS_SPEED_VALS = [
                "30MHz",
                "20MHz",
                "10MHz",
                "1MHz",
            ]
        else:
            HS_SPEED_VALS = ["1MHz", "0.1MHz"]
        return HS_SPEED_VALS[self.hs_speed_index]

    def get_vs_speed(self):
        VS_SPEED_VALS = ["0.6us", "1.12us", "2.2us", "4.3us"]
        return VS_SPEED_VALS[self.vs_speed_index]

    def get_sensitivity(self):
        if self.amp_index is None:
            return None

        if self.amp_index == 0:
            SENSITIVITY_VALS = [
                18.2,
                5.54,
                16.4,
                4.47,
                16.3,
                4.00,
                16.1,
                3.89,
            ]
        else:
            SENSITIVITY_VALS = [3.33, 0.8, 3.34, 0.8]
        return SENSITIVITY_VALS[2 * self.hs_speed_index + self.preamp_gain_index]

    def get_dict(self):
        return {
            'exposure_time': self.exposure_time,
            'preamp_gain_index': self.preamp_gain_index,
            'hs_speed_index': self.hs_speed_index,
            'vs_speed_index': self.vs_speed_index,
            'em_gain': self.em_gain,
            'em_advanced': self.em_advanced,
            'em_gain_mode': self.em_gain_mode,
            'acquisition_mode': self.acquisition_mode,
            'num_accumulations': self.num_accumulations,
            'readout_mode': self.readout_mode,
            'num_kinetics': self.num_kinetics,
            'kinetic_cycle_time': self.kinetic_cycle_time,
            'temperature': self.temperature,
            'trigger_mode': self.trigger_mode,
            'amp_index': self.amp_index,
            'baseline_clamp': self.baseline_clamp,
            'cameralink_mode': self.cameralink_mode,
            'frametransfer_mode': self.frametransfer_mode,
            'max_path': self.max_path,
        }
