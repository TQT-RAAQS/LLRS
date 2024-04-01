#ifndef LLRS_LIB_SETTINGS_H_
#define LLRS_LIB_SETTINGS_H_

// Global Checkers
#define LLRS_OK 0
#define LLRS_ERR 1

// AWG Settings
#define AWG_NUM_SEGMENTS int(256) //# of segments that AWG memory is split into
#define AWG_SAMPLE_RATE double(624e6) // Hz
#define TRIGGER_SIZE                                                           \
    size_t(2000)        //# of Waveforms repeated for the synchronized trigger
#define WFM_MASK 0x7FFF // == 1 << 15 - 1 == 0111 1111 1111 1111
#define VPP size_t(280) // peak to peak voltage in mV

// Waveform Settings
#define WAVEFORM_DUR double(100e-6) // Seconds
#define WAVEFORM_LEN                                                           \
    size_t(AWG_SAMPLE_RATE *WAVEFORM_DUR) // number of samples per waveform
                                          // (sample in shorts)
#define WF_PER_SEG int(32)
#define NULL_LEN int(62400) // number of samples in null segment
#define IDLE_LEN int(62400) // number of samples in idle segment

// Image Acquisition
#define MAX_IMAGE_READ_BUFFER_SIZE size_t(1048576) // 1024 * 1024, unit: #pixels
#define ACQ_TIMEOUT size_t(600)                    // ms
#define ASYNC_TRIG_AMP size_t(3)                   // volts

// Displacements
#define DISPLACEMENT_RELATIVE_SLOPE 3e4
#define DISPLACEMENT_FREQUENCY_RELATIVE_SLOPE 2.5e10

// Image Processing
#define FILTERING_NUM_THREADS size_t(32)
#define IMAGE_INVERTED_X true

// Number of cycles to attempt before existing a trial
#define ATTEMPT_LIMIT size_t(5)

enum Channel { CHAN_0 = 0b00, CHAN_1 = 0b01, CHAN_2 = 0b10, CHAN_3 = 0b11 };

enum WfType {
    STATIC = 0b0000,
    EXTRACT = 0b1000,
    IMPLANT = 0b1010,
    FORWARD = 0b1100,
    BACKWARD = 0b1110,
    RIGHTWARD = 0b1101,
    LEFTWARD = 0b1111,
    NULL_WF = 0b0001
};

#define LOAD_EFFICIENCY 0.6

#endif
