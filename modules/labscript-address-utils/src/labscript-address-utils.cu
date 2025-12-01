#include "labscript-address-utils.h"

std::string LabscriptAddressUtils::get_experiment_folder_name(std::string shot_address) {
    boost::filesystem::path shot_path = shot_address;
    boost::filesystem::path output = shot_path.parent_path().parent_path();
    return output.string();
}

std::string LabscriptAddressUtils::get_images_folder_name(std::string shot_address, std::string image_folder_name) {
    boost::filesystem::path output = LabscriptAddressUtils::get_experiment_folder_name(shot_address);
    output = output / boost::filesystem::path(image_folder_name);
    output = output / boost::filesystem::path(shot_address).filename();

    std::string output_str = output.string();
    return output_str.substr(0, output_str.length() - 3);
}
