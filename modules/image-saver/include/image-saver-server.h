#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <mutex>
#include <boost/filesystem.hpp>
#include <tuple>
#include <thread>
#include "server.hpp"
#include <yaml-cpp/yaml.h>
#include "llrs-lib/PreProc.h"
#include "emccd-config.h"
#include "shot-file.h"
#include "activesilicon-1xcld.hpp"

using ImageBatch = std::tuple<std::vector<uint16_t>, std::string>; // Image, file address

class ImageSaverServer {

    YAML::Node config;
    
    std::unique_ptr<ActiveSilicon1XCLD> fgc;
    YAML::Node config_fgc;
    int fgc_timeout_ms;
    
    std::unique_ptr<Server> server;
    int port;
    int listen_timeout;
    std::string image_folder_name;

    std::string experiment_folder = "";

    std::atomic<bool> flag_thread_running;
    
    std::mutex cache_mutex; 
    int image_counter = 0;
    std::string image_folder_address = "";
    int timestamp = 0;
    std::vector<ImageBatch> images_cache;

    std::string handle_request(std::string request);
    void transition_to_buffered(std::string h5_address);
    void transition_to_static();
    void configure_fgc(std::string shot_address);

    std::string get_images_folder_name(std::string shot_address);
    std::string get_experiment_folder_name(std::string shot_address);

    void capture_images();

protected:

    void setup_server();
    void setup_fgc();
    void setup_image_capturer_thread();

public:

    ImageSaverServer(std::string config_str);
    void start_server();

};