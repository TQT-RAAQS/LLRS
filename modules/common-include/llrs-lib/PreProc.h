#ifndef LLRS_LIB_PRE_PROC_H_
#define LLRS_LIB_PRE_PROC_H_

#include "log.h"
#include <cstdlib>
#include <experimental/filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

// Define separator based on the platform
#ifndef _K_PATH_SEPARATOR
#define _K_PATH_SEPARATOR
#ifdef _WIN32
const char kPathSeparator1 = '\\';
#else
const char kPathSeparator1 = '/';
#endif
#endif

namespace fs = std::experimental::filesystem;

#define FILE_EXISTS(name) (fs::exists(name))

#define PSF_PATH(fname)                                                        \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
     kPathSeparator1 + "psf" + kPathSeparator1 + (fname))
#define COEF_X_PATH(fname)                                                     \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
     kPathSeparator1 + "coef" + kPathSeparator1 + "primary" +                  \
     kPathSeparator1 + (fname))
#define COEF_Y_PATH(fname)                                                     \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
     kPathSeparator1 + "coef" + kPathSeparator1 + "secondary" +                \
     kPathSeparator1 + (fname))
#define WF_REPO_PATH(fname)                                                    \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
     kPathSeparator1 + "wfm" + kPathSeparator1 + (fname))
#define LOGGING_PATH(fname)                                                    \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
     kPathSeparator1 + "runtime" + kPathSeparator1 + "logs" +                  \
     kPathSeparator1 + (fname))

#define CONFIG_PATH(id)                                                        \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "configs" +        \
     kPathSeparator1 + "llrs" + kPathSeparator1 + (id))
#define SOLN_PATH(id)                                                          \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
     kPathSeparator1 + "runtime-benchmark-solutions" + kPathSeparator1 + (id) +      \
     ".json")
#define BENCHMARK_PATH(id)                                                     \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
     kPathSeparator1 + "runtime-benchmark-data" + kPathSeparator1 + (id) +         \
     ".json")
#define IMAGE_PATH(epoch)                                                      \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
     kPathSeparator1 + "images" + kPathSeparator1 + (epoch) + ".pgm")

#define TRIAL_NAME(num) ("trial_" + std::to_string(num))
#define REP_NAME(num) ("repetition_" + std::to_string(num))
#define CYCLE_NAME(num) ("cycle_" + std::to_string(num))

#endif
