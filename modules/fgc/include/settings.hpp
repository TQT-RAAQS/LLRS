// Logging Definitions
#ifndef SETTINGS_HPP
#define SETTINGS_HPP

// Global Checkers
#define CAMERA_OK 0
#define CAMERA_ERR 1

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

#endif
