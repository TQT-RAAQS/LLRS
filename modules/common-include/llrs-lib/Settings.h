#ifndef LLRS_LIB_SETTINGS_H_
#define LLRS_LIB_SETTINGS_H_

// Global Checkers
#define LLRS_OK 0
#define LLRS_ERR 1

// Image Acquisition
#define MAX_IMAGE_READ_BUFFER_SIZE size_t(1048576) // 1024 * 1024, unit: #pixels

// Image Processing
#define FILTERING_NUM_THREADS size_t(32)
#define IMAGE_INVERTED_X true

// Number of cycles to attempt before exiting a trial
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

#endif
