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

#define LLRS_RESET_PATH (std::string("") + PROJECT_BASE_DIR + "/configs/llrs/config.json")
#define PSF_CONFIG_PATH (std::string("") + PROJECT_BASE_DIR + "/tools/psf_translator.py")

#endif
