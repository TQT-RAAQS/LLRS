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

#define EXPERIMENTS_ROOT_DIR                                                   \
    (std::string("") + SHARED_DRIVE_DIR + kPathSeparator1 + "Experiments" +    \
     kPathSeparator1 + "Rydberg")
#define PSF_PATH(fname)                                                        \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
     kPathSeparator1 + "psf" + kPathSeparator1 + (fname))
#define PSF_TRANSLATION_FILE                                                   \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
      kPathSeparator1 + "psf" + kPathSeparator1 + "psfs.bin")
#define TRAPS_ORDERS_TRANSLATION_FILE                                          \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
       kPathSeparator1 + "psf" + kPathSeparator1 + "orders.bin")
#define CONFIGS_PSF_TRANSLATION_READY_FILE                                     \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
       kPathSeparator1 + "translation" + kPathSeparator1 + "psf_flag.done")
#define IQMIXER_TRANSLATION_FILE                                               \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
        kPathSeparator1 + "iqmixer" + kPathSeparator1 + "iqmixer.bin")
#define IQMIXER_TRANSLATION_READY_FILE                                         \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
        kPathSeparator1 + "iqmixer" + kPathSeparator1 + "iqmixer.done")
#define CONFIGS_TRANSLATOR_SCRIPT                                              \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "tools" +          \
     kPathSeparator1 + "config-translator.py")
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
     kPathSeparator1 + "logs" + kPathSeparator1 + (fname))

#define CONFIG_PATH(id)                                                        \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "configs" +        \
     kPathSeparator1 + "llrs" + kPathSeparator1 + (id))
#define WFM_CONFIG_PATH(id)                                                    \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "configs" +        \
     kPathSeparator1 + "waveforms" + kPathSeparator1 + (id))
#define POWER_SAFETY_CONFIG_PATH(id)                                           \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "configs" +        \
     kPathSeparator1 + "waveform-power-safety" + kPathSeparator1 + (id))
#define SOLN_PATH(id)                                                          \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
     kPathSeparator1 + "runtime-benchmark-solutions" + kPathSeparator1 +       \
     (id) + ".json")
#define BENCHMARK_PATH(id)                                                     \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
     kPathSeparator1 + "runtime-benchmark-data" + kPathSeparator1 + (id) +     \
     ".json")
#define IMAGE_PATH(epoch)                                                      \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "resources" +      \
     kPathSeparator1 + "images" + kPathSeparator1 + (epoch) + ".pgm")
#define MICROWAVE_AWG_HANDLER_CONFIG(id)                                       \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "configs" +        \
    kPathSeparator1 + "microwave-handler" + kPathSeparator1 + (id))
#define IMAGE_SAVER_SERVER(id)                                                 \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "configs" +        \
     kPathSeparator1 + "image-saver-server" + kPathSeparator1 + (id))
#define IMAGE_SAVER_FGC(id)                                                    \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "configs" +        \
     kPathSeparator1 + "image-saver-fgc" + kPathSeparator1 + (id))
#define TRAP_RESULT_SAVER(id)                                                  \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "configs" +        \
    kPathSeparator1 + "trap-result-saver" + kPathSeparator1 + (id))
#define SHARED_MEMORY_HANDLER(id)                                              \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "configs" +        \
      kPathSeparator1 + "shared-memory-handler" + kPathSeparator1 + (id))
#define MASTER_SHARED_MEMORY_HANDLER_SERVER(id)                                \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "configs" +        \
      kPathSeparator1 + "master-shared-memory-handler-server" + kPathSeparator1 + (id))
#define RAMSEY_STABILIZER(id)                                                  \
    (std::string("") + PROJECT_BASE_DIR + kPathSeparator1 + "configs" +        \
      kPathSeparator1 + "ramsey-stabilizer" + kPathSeparator1 + (id))

#define TRIAL_NAME(num) ("trial_" + std::to_string(num))
#define REP_NAME(num) ("repetition_" + std::to_string(num))
#define CYCLE_NAME(num) ("cycle_" + std::to_string(num))

#endif
