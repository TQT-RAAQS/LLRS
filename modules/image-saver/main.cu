#include "image-saver-server.h"

int main() {
    // ImageSaverServer iss("default.yml");
    // iss.start_server();

    // ACQUIRING IMAGES
    ActiveSilicon1XCLD fgc_object;
    ActiveSilicon1XCLD* fgc = &fgc_object;

    int roi_width = 1024, roi_height = 1024, roi_xoffset = 0, roi_yoffset = 0, roi_vbin = 1, roi_hbin = 1;
    double fgc_timeout_ms = 3000;

    fgc->setup(roi_width, roi_height, fgc_timeout_ms, roi_xoffset, roi_yoffset, roi_vbin, roi_hbin);
    std::vector<uint16_t> current_image = fgc->acquire_single_image();
    fgc->destroy_handle();
    fgc = new ActiveSilicon1XCLD();
    fgc->setup(roi_width, roi_height, fgc_timeout_ms, roi_xoffset, roi_yoffset, roi_vbin, roi_hbin);
    current_image = fgc->acquire_single_image();
    fgc->destroy_handle();
    fgc = new ActiveSilicon1XCLD();
    fgc->setup(roi_width, roi_height, fgc_timeout_ms, roi_xoffset, roi_yoffset, roi_vbin, roi_hbin);
    current_image = fgc->acquire_single_image();
    fgc->destroy_handle();
    fgc = new ActiveSilicon1XCLD();
    fgc->setup(roi_width, roi_height, fgc_timeout_ms, roi_xoffset, roi_yoffset, roi_vbin, roi_hbin);
    current_image = fgc->acquire_single_image();
    fgc->destroy_handle();
    fgc = new ActiveSilicon1XCLD();
    fgc->setup(roi_width, roi_height, fgc_timeout_ms, roi_xoffset, roi_yoffset, roi_vbin, roi_hbin);
    current_image = fgc->acquire_single_image();
    std::cout << current_image.size() << std::endl;



    // READING SHOTS
    // std::string address = "/home/tqtraaqs1/Z/Experiments/Rydberg/2024-12-17/01_55_09-test/labscript_shot_outputs/test_2024-12-17_0000_00.h5";

    // ShotFile shot(address);
    // EmccdConfig emccd_config(shot);

    // std::cout << emccd_config.get_hbin() << std::endl;



    // ZMQ Server
    // std::string config_address = IMAGE_SAVER_SERVER("default.yml");
    // YAML::Node node = YAML::LoadFile(config_address);
    // int port = node["port"].as<int>();
    // int listen_timeout = node["listen_timeout"].as<int>();

    // std::string request;
    // Server server(port, listen_timeout);
    // server.listen(request);
    
}