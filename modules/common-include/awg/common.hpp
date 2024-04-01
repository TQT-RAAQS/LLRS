#ifndef AWG_COMMON_HPP_
#define AWG_COMMON_HPP_

// Common STD Includes
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <experimental/filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>

// Define separator based on the platform
#ifdef _WIN32
const char kPathSeparator = '\\';
#else
const char kPathSeparator = '/';
#endif

namespace fs = std::experimental::filesystem;

#define FILE_EXISTS(name) (fs::exists(name))

#define AWG_OK 0
#define AWG_ERR 1

#define AWG_CONFIG_PATH                                                        \
    (std::string("") + PROJECT_BASE_DIR + "config/awg/awg.yml")

#endif
