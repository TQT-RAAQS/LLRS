#ifndef LLCS_COMMON_HPP_
#define LLCS_COMMON_HPP_

// Common STD Includes
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <experimental/filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

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
    ST_IDLE,
    ST_RESET,
    ST_CONFIG_PSF,
    ST_CONFIG_WAVEFORM,
    ST_PROCESS_SHOT,
    ST_READY,
    ST_LLRS_EXEC,
    ST_FAULT,
    ST_EXIT,
    ST_TRIGGER_DONE,
    ST_CLOSE_AWG,
    ST_RESTART_AWG
};

enum TransitionType { TR_FAULT, TR_HW, TR_NW, TR_INTERNAL, TR_NULL };

enum NWTransition { CONFIG_HW, CONFIG_SM, READY, DONE };

enum ModuleType { M_LLRS, M_CLO, M_RYDBERG };

// Hardware Trigger types
#define NO_HW_TRIG -1
#define HW_TRIG1 1
#define HW_TRIG2 2
#define HW_TRIG3 3
#define HW_TRIG4 4

// Return types
#define SYS_OK 0
#define SYS_ERR 1

#define PSF_TRANSLATOR_PATH (std::string("") + PROJECT_BASE_DIR + "/tools/psf-translator.py")

#endif
