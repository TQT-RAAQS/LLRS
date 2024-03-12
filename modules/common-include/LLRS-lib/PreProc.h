/**
 * @brief Defines macros for the C preprocessor
 * @date Jan 2024
*/

#ifndef _PRE_PROC_H_
#define _PRE_PROC_H_

#include <cstdlib>
#include <experimental/filesystem>
#include <sstream>
#include <iostream>
#include <iomanip>

// Define separator based on the platform
#ifndef _K_PATH_SEPARATOR
#define _K_PATH_SEPARATOR
    #ifdef _WIN32
const char kPathSeparator1 = '\\';
#   else
const char kPathSeparator1 = '/';
#   endif
#endif

namespace fs = std::experimental::filesystem;

#define FILE_EXISTS(name) (fs::exists(name))
std::string NADA = "";

#define PSF_PATH(fname) (NADA + PROJECT_BASE_DIR + kPathSeparator1 + "resources" + kPathSeparator1 + "psf" + kPathSeparator1 + (fname))
#define COEF_X_PATH(fname) (NADA + PROJECT_BASE_DIR + kPathSeparator1 + "resources" + kPathSeparator1 + "coef" + kPathSeparator1 + "primary" + kPathSeparator1 + (fname))
#define COEF_Y_PATH(fname) (NADA + PROJECT_BASE_DIR + kPathSeparator1 + "resources" + kPathSeparator1 + "coef" + kPathSeparator1 + "secondary" + kPathSeparator1 + (fname))
#define WF_REPO_PATH(fname) (NADA + PROJECT_BASE_DIR + kPathSeparator1 + "resources" + kPathSeparator1 + "wfm" + kPathSeparator1 + (fname))
#define LOGGING_PATH(fname) (NADA + PROJECT_BASE_DIR + kPathSeparator1 + "resources" + kPathSeparator1 + "logs" + kPathSeparator1 + (fname))

#define CONFIG_PATH(id) (NADA + PROJECT_BASE_DIR + kPathSeparator1 + "configs" + kPathSeparator1 + "LLRS" + kPathSeparator1 + (id))
#define SOLN_PATH(id) (NADA + PROJECT_BASE_DIR + kPathSeparator1 + "resources" + kPathSeparator1 + "operational_solutions" + kPathSeparator1 + (id) + ".json")
#define BENCHMARK_PATH(id) (NADA + PROJECT_BASE_DIR + kPathSeparator1 + "resources" + kPathSeparator1 + "runtime_benchmarks" + kPathSeparator1 + (id) + ".json")
#define IMAGE_PATH(epoch) (NADA + PROJECT_BASE_DIR + kPathSeparator1 + "resources" + kPathSeparator1 + "images" + kPathSeparator1 + (epoch) + ".pgm")

#define TRIAL_NAME(num) ("trial_" + std::to_string(num))
#define REP_NAME(num) ("repetition_" + std::to_string(num))
#define CYCLE_NAME(num) ("cycle_" + std::to_string(num))

#define __FILENAME__ (__builtin_strrchr(__FILE__, '/') ? __builtin_strrchr(__FILE__, '/') + 1 : __FILE__)    // only show filename and not it's path (less clutter)

static std::time_t time_now = std::time(nullptr);

#define INFO \
    time_now = std::time(nullptr); \
    std::clog << std::put_time(std::localtime(&time_now), "%y-%m-%d %OH:%OM:%OS") << " [INFO] " << __FILENAME__ << "(" << __FUNCTION__ << ":" << __LINE__ << ") >> "

#define ERROR \
    time_now = std::time(nullptr); \
    std::clog << std::put_time(std::localtime(&time_now), "%y-%m-%d %OH:%OM:%OS") << " [ERROR] " << __FILENAME__ << "(" << __FUNCTION__ << ":" << __LINE__ << ") >> "


#endif