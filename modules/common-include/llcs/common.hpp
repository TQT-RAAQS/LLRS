#ifndef COMMON_HPP
#define COMMON_HPP

// Common STD Includes
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <functional>
#include <cstring>
#include <cmath>
#include <csignal>
#include <atomic>
#include <tuple>
#include <algorithm>
#include <experimental/filesystem>
#include <sstream>

// Define separator based on the platform
#ifdef _WIN32
const char kPathSeparator2 = '\\';
#else
const char kPathSeparator2 = '/';
#endif

namespace fs = std::experimental::filesystem;

// Type Definitions
enum StateType {
    ST_BEGIN,
    ST_FAULT,
    ST_IDLE,
    ST_CONFIG_HW,
    ST_CONFIG_SM,
    ST_READY,
    ST_EXPERIMENT,
    ST_EXIT,
    ST_RESET,
    ST_TRIGGER_DONE,
    ST_LAST_TRIGGER_DONE,
    ST_CLOSE_AWG,
    ST_RESTART_AWG,
    ST_LLRS_EXEC,
    ST_CLO_EXEC,
    ST_RYDBERG_EXEC
};

enum TransitionType {
    TR_FAULT,
    TR_HW,
    TR_NW,
    TR_INTERNAL,
    TR_NULL
};

enum NWTransition{
    CONFIG_HW,
    CONFIG_SM,
    READY,
    DONE
};

enum ModuleType {
    M_LLRS,
    M_CLO,
    M_RYDBERG
};

// AWG settings
#define AWG_NUM_SEGMENTS            int (256)                              //# of segments that AWG memory is split into
#define AWG_SAMPLE_RATE             double (624e6)                         //Hz
#define TRIGGER_SIZE                size_t (2000)                          //# of Waveforms repeated for the synchronized trigger
#define WFM_MASK                    0x7FFF                                 // == 1 << 15 - 1 == 0111 1111 1111 1111
#define VPP                         size_t (80)                            //peak to peak voltage in mV

//Waveform Settings
#define WAVEFORM_DUR                double (100e-6)                         //Seconds
#define WAVEFORM_LEN                size_t (AWG_SAMPLE_RATE * WAVEFORM_DUR)//number of samples per waveform (sample in shorts)
#define WF_PER_SEG                  int (32)
#define NULL_LEN                    int (6240)                             //number of samples in null segment
#define IDLE_LEN                    int (6240)

// Hardware Trigger types
#define NO_HW_TRIG      -1
#define HW_TRIG1        1
#define HW_TRIG2        2
#define HW_TRIG3        3
#define HW_TRIG4        4

// Return types
#define SYS_OK          0
#define SYS_ERR         1

#define LLRS_RESET_PATH ()

#define PSF_CONFIG_PATH ()


#endif