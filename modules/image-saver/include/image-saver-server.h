#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include "server.hpp"
#include <yaml-cpp/yaml.h>
#include "llrs-lib/PreProc.h"
#include "emccd-config.h"
#include "shot-file.h"
#include "activesilicon-1xcld.hpp"

class ImageSaverServer {

    // std::atomic<bool> flag_acquire_image(true);
    YAML::Node config;
    int port;
    int listen_timeout;
    std::unique_ptr<Server> server;

    void stop_acquiring_images();
    std::string handle_request(std::string request);
    void transition_to_buffered(std::string h5_address);
    void transition_to_static();

public:

    ImageSaverServer(std::string config_str);
    void start_server();

};