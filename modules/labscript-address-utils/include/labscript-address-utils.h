#ifndef LABSCRIPT_ADDRESS_UTILS_H_
#define LABSCRIPT_ADDRESS_UTILS_H_

#include <string>
#include <boost/filesystem.hpp>

namespace LabscriptAddressUtils {

    std::string get_experiment_folder_name(std::string shot_address);
    std::string get_images_folder_name(std::string shot_address, std::string image_folder_name);
}

#endif