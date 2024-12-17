#include "image-saver-server.h"

ImageSaverServer::ImageSaverServer(std::string config_str) {
    std::string config_address = IMAGE_SAVER_SERVER(config_str);
    config = YAML::LoadFile(config_address);

    setup_server();
    setup_fgc();
    setup_image_capturer_thread();
}

void ImageSaverServer::setup_server() {
    port = config["port"].as<int>();
    listen_timeout = config["listen_timeout"].as<int>();
    image_folder_name = config["image_folder_name"].as<std::string>();
    server = std::make_unique<Server>(port, listen_timeout);
}

void ImageSaverServer::setup_fgc() {
    std::string fgc_config_address = IMAGE_SAVER_FGC(config["fgc_config"].as<std::string>());
    config_fgc = YAML::LoadFile(fgc_config_address);

    fgc = std::make_unique<ActiveSilicon1XCLD>();
    fgc_timeout_ms = config_fgc["timeout"].as<int>();
    flag_thread_running.store(false);
}

void ImageSaverServer::capture_images() {
    while (true) {
        if (flag_thread_running.load()) {
            std::vector<uint16_t> current_image = fgc->acquire_single_image();
            if (current_image.size() == 0) {
                flag_thread_running.store(false);
            } else {
                std::lock_guard<std::mutex> lock(cache_mutex);

                std::ostringstream filename_stream;
                filename_stream << "image_" << timestamp << "_" << image_counter << ".png";
                std::string file_name = filename_stream.str();
                images_cache.push_back(ImageBatch(
                    current_image,
                    (boost::filesystem::path(image_folder_address) / boost::filesystem::path(file_name)).string()
                ));
                image_counter++;
            }
        }
    }
}

void ImageSaverServer::setup_image_capturer_thread() {
    flag_thread_running.store(false);
    std::thread image_capturer_thread(&ImageSaverServer::capture_images, this);
    image_capturer_thread.detach();
}

void ImageSaverServer::start_server() {
    INFO << "Image Saver Server initialized." << std::endl;

    std::string request, response;
    while (true) {
        if (server->listen(request) == 1) {
            break; // Exception occured during listening/parsing the request, e.g. listening timeout. Error logged by zmq.
        }

        try {
            INFO << "Handling request: " << request << std::endl;
            response = handle_request(request);
            server->send(response);

            if (request == "exit") {
                break;
            }
        } catch (const std::exception &e) {
            std::cerr << "Error when handling the request: " << request << "; " << e.what() << std::endl;
        }
    }
}

std::string ImageSaverServer::handle_request(std::string request) {
    if (request == "hello") {
        return "hello";
    }
    if (request == "exit") {
        std::cout << images_cache.size() << std::endl;
        return "200";
    }
    if (request == "abort") {
        return "done";
    }
    if (request == "done") {
        server->send("ok");
        std::string empty_string;
        server->listen(empty_string);
        transition_to_static();
        return "done";
    }
    if (request.substr(request.size() - 3) == ".h5") {
        server->send("ok");
        std::string empty_string;
        server->listen(empty_string);
        transition_to_buffered(request);
        return "done";
    }

    return "404";
}

std::string ImageSaverServer::get_experiment_folder_name(std::string shot_address) {
    boost::filesystem::path shot_path = shot_address;
    boost::filesystem::path output = shot_path.parent_path().parent_path();
    return output.string();
}

std::string ImageSaverServer::get_images_folder_name(std::string shot_address) {
    boost::filesystem::path output = get_experiment_folder_name(shot_address);
    output = output / boost::filesystem::path(image_folder_name);
    output = output / boost::filesystem::path(shot_address).filename();

    std::string output_str = output.string();
    return output_str.substr(0, output_str.length() - 3);
}

void ImageSaverServer::transition_to_buffered(std::string h5_address) {
    std::string adjusted_h5_address = adjust_address(h5_address); // Converting server address to local address on this workstation
    INFO << "Processing a new shot: " << adjusted_h5_address << std::endl;
    
    std::string new_experiment_folder = get_experiment_folder_name(adjusted_h5_address);
    if (experiment_folder != new_experiment_folder || !flag_thread_running.load()) {
        experiment_folder = new_experiment_folder;
        configure_fgc(adjusted_h5_address);
    }

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        image_folder_address = get_images_folder_name(adjusted_h5_address);
        timestamp = int(std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count());
        image_counter = 0;
    }
}

void ImageSaverServer::configure_fgc(std::string h5_address) {
    INFO << "Configuring the FGC" << std::endl;

    while (flag_thread_running.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    ShotFile shot(h5_address);
    EmccdConfig emccd_config(shot);

    fgc->destroy_handle();
    setup_fgc();
    fgc->setup(
        emccd_config.get_roi_w(),
        emccd_config.get_roi_h(),
        fgc_timeout_ms, 
        emccd_config.get_roi_x(),
        emccd_config.get_roi_y(), 
        emccd_config.get_vbin(), 
        emccd_config.get_hbin()
    );

    flag_thread_running.store(true);
}

void ImageSaverServer::transition_to_static() {
    INFO << "Transitioning to static" << std::endl;

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        INFO << "Total images captured in this shot: " << image_counter << std::endl;
    }
}