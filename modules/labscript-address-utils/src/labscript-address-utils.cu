#include "labscript-address-utils.h"

std::string LabscriptAddressUtils::get_experiment_folder_path(std::string shot_address) {
    boost::filesystem::path shot_path = shot_address;
    boost::filesystem::path output = shot_path.parent_path().parent_path();
    return output.string();
}

std::string LabscriptAddressUtils::get_experiment_folder_name(std::string shot_address) {
    boost::filesystem::path shot_path = shot_address;
    boost::filesystem::path output = shot_path.parent_path().parent_path().filename();
    return output.string();
}

std::string LabscriptAddressUtils::get_images_folder_name(std::string shot_address, std::string image_folder_name) {
    boost::filesystem::path output = LabscriptAddressUtils::get_experiment_folder_path(shot_address);
    output = output / boost::filesystem::path(image_folder_name);
    output = output / boost::filesystem::path(shot_address).filename();

    std::string output_str = output.string();
    return output_str.substr(0, output_str.length() - 3);
}

std::string LabscriptAddressUtils::get_experiment_directory_from_time(std::string date, std::string time_stamp) {
    boost::filesystem::path output = EXPERIMENTS_ROOT_DIR;
    output = output / boost::filesystem::path(date);

    if (!boost::filesystem::exists(output) || !boost::filesystem::is_directory(output)) {
        throw std::runtime_error("Could not find the experiment with the provided date: " + output.string());
    }

    for (auto const& entry : boost::filesystem::directory_iterator(output)) {
        if (boost::filesystem::is_directory(entry)) {
            auto name = entry.path().filename().string();

            if (name.rfind(time_stamp, 0) == 0) {
                output = entry.path();
                return output.string();
            }
        }
    }

    throw std::runtime_error("Could not find the experiment with the provided time stamp: " + output.string());
}