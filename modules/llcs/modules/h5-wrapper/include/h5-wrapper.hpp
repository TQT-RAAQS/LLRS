#ifndef H5_WRAPPER_HPP_
#define H5_WRAPPER_HPP_

#include "llcs/common.hpp"
#include <cstdlib>
#include <fstream>
#include <hdf5/serial/H5Cpp.h>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

namespace H5Wrapper {

std::map<std::string, std::vector<ModuleType>>
parseSMConfig(std::string filepath);
std::string convertHWConfig(std::string filepath);

}; // namespace H5Wrapper
#endif
