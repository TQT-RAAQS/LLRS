#ifndef LABSCRIPT_ADDRESS_UTILS_H_
#define LABSCRIPT_ADDRESS_UTILS_H_

#include <string>
#include <boost/filesystem.hpp>
#include <llrs-lib/PreProc.h>

namespace LabscriptAddressUtils {

    std::string get_experiment_folder_path(std::string shot_address);
    std::string get_experiment_folder_name(std::string shot_address);
    std::string get_images_folder_name(std::string shot_address, std::string image_folder_name);
    
    std::string get_experiment_directory_from_time(std::string date, std::string time_stamp);
}

#endif