#ifndef COMMON_HPP
#define COMMON_HPP

// Common STD Includes
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <functional>
#include <cstring>
#include <cmath>
#include <csignal>
#include <atomic>
#include <yaml-cpp/yaml.h>
#include <cstdlib>
#include <experimental/filesystem>
#include <sstream>

// Define separator based on the platform
#ifdef _WIN32
const char kPathSeparator = '\\';
#else
const char kPathSeparator = '/';
#endif

namespace fs = std::experimental::filesystem;

#define FILE_EXISTS(name) (fs::exists(name))


#define AWG_OK      0
#define AWG_ERR     1

#define AWG_CONFIG_PATH (std::string("")  + PROJECT_BASE_DIR + "config/awg/awg.yml")

#endif
