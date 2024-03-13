#include "common.hpp"
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>
#include <hdf5/serial/H5Cpp.h>
#include <fstream>
#include <cstdlib>

namespace H5Wrapper {

    std::map<std::string, std::vector<ModuleType>> parseSMConfig(std::string filepath);
    std::string convertHWConfig(std::string filepath);
    
};
