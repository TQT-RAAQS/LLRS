#ifndef IMAGE_SAVER_SERVER_
#define IMAGE_SAVER_SERVER_

#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <mutex>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
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
    int roi_w, roi_h;
    
    std::unique_ptr<Server> server;
    int port;
    int listen_timeout;
    std::string image_folder_name;

    std::string previous_experiment_folder = "";
    std::string experiment_folder = "";

    std::atomic<bool> flag_thread_running;
    std::atomic<bool> flag_thread_killed;

    std::thread image_saver_thread;
    std::thread image_capturer_thread;
    
    std::mutex cache_mutex;
    int image_counter = 0;
    std::string image_folder_address = "";
    long timestamp = 0;
    std::vector<ImageBatch> images_cache;

    std::string handle_request(std::string request);
    void transition_to_buffered(std::string h5_address);
    void transition_to_static();
    void configure_fgc(std::string shot_address);
    void set_fgc_roi(int roi_w, int roi_h, int timeout_ms, int roi_x, int roi_y, int vbin, int hbin);

    void capture_images();
    void save_images();
    
    static std::string get_images_folder_name(std::string shot_address, std::string image_folder_name);
    static std::string get_experiment_folder_name(std::string shot_address);
    static void create_directory(boost::filesystem::path path);

protected:

    void setup_zmq_client();
    void setup_fgc();
    void setup_image_capturer_thread();
    void setup_saver_worker();

public:

    ImageSaverServer(std::string config_str);
    void start_server();

};

#endif