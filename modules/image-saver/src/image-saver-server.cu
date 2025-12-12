#include "image-saver-server.h"

/************************************************************************************************** */

void ImageSaverServer::create_directory(boost::filesystem::path path) {
    boost::filesystem::path parent = path.parent_path();
    if (!boost::filesystem::exists(parent)) {
        ImageSaverServer::create_directory(parent);
    }
    boost::filesystem::create_directories(path.string());
}

/************************************************************************************************** */

void ImageSaverServer::start_server() {
    INFO << "Image Saver Server initialized." << std::endl;

    std::string request, response;
    while (true) {
        if (server->listen(request) == 1) {
            break; // In case exception occured when  listening/parsing the request, e.g. listening timeout. Error logged by zmq.
        }

        try {
            INFO << "Handling request: " << request << std::endl;
            if (request == "exit") {
                server->send("200");
                break;
            }

            response = handle_request(request);
            server->send(response);
        } catch (const std::exception &e) {
            std::cerr << "Error when handling the request: " << request << "; " << e.what() << std::endl;
        }
    }

    this->flag_thread_killed.store(true);
    if (this->image_capturer_thread.joinable()) this->image_capturer_thread.join();
    if (this->image_saver_thread.joinable()) this->image_saver_thread.join();
}

std::string ImageSaverServer::handle_request(std::string request) {
    if (request == "hello") {
        return "hello";
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

void ImageSaverServer::transition_to_buffered(std::string h5_address) {
    std::string adjusted_h5_address = adjust_address(h5_address); // Converting server address to local address on this workstation
    INFO << "Processing a new shot: " << adjusted_h5_address << std::endl;
    
    std::string new_experiment_folder = LabscriptAddressUtils::get_experiment_folder_path(adjusted_h5_address);
    if (experiment_folder != new_experiment_folder || !flag_thread_running.load()) {
        experiment_folder = new_experiment_folder;
        configure_fgc(adjusted_h5_address);
        this->reload_psf_data();
    }
    
    this->shared_memory_handler->change_shot_address(adjusted_h5_address); // Update the shot address
    this->shared_memory_handler->signal_done(); // Tell master that a new shot is about to start.

    this->shared_memory_handler->wait_for_update(); // Wait until master confims all the worker processes are done.

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        image_folder_address = LabscriptAddressUtils::get_images_folder_name(adjusted_h5_address, image_folder_name);
        timestamp = long(std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count() * 1000);
        image_counter = 0;
    }
}

void ImageSaverServer::transition_to_static() {
    INFO << "Transitioning to static" << std::endl;

    {
        this->shared_memory_handler->signal_done(); // Signal master that the shot is over.
        this->shared_memory_handler->wait_for_update(); // Wait for master until it confirms processing of the shot is done.
        this->shared_memory_handler->reset_image_count(); // Resetting the number of images to 0 on the shared memory.
        
        std::lock_guard<std::mutex> lock(cache_mutex);
        INFO << "Total images captured in this shot: " << image_counter << std::endl;

        std::string address = (boost::filesystem::path(image_folder_address) / boost::filesystem::path("emccd_iss.done")).string();
        std::vector<uint16_t> empty_image;
        std::cout << "address: " << address << std::endl;
        images_cache.push_back(ImageBatch(empty_image, address));
    }
}

void ImageSaverServer::configure_fgc(std::string h5_address) {
    INFO << "Configuring the FGC" << std::endl;

    // TODO: The destroy handle function breaks the FGC. Fix the bugs in FGC.
    // while (flag_thread_running.load()) {
    // }

    // ShotFile shot(h5_address);
    // EmccdConfig emccd_config(shot);

    // fgc->destroy_handle(); # Suppoed to close connection
    // set_fgc_roi(
    //     emccd_config.get_roi_w(),
    //     emccd_config.get_roi_h(),
    //     fgc_timeout_ms, 
    //     emccd_config.get_roi_x(),
    //     emccd_config.get_roi_y(), 
    //     emccd_config.get_vbin(), 
    //     emccd_config.get_hbin()
    // );

    flag_thread_running.store(true);
}

/************************************************************************************************** */

ImageSaverServer::ImageSaverServer(std::string config_str) {
    std::string config_address = IMAGE_SAVER_SERVER(config_str);
    INFO << config_address << std::endl;
    config = YAML::LoadFile(config_address);
    this->flag_thread_killed.store(false);

    setup_zmq_client();
    setup_fgc();
    setup_image_capturer_thread();
    setup_saver_worker();
    reload_psf_data();
    setup_shared_memory_handler();
}

ImageSaverServer::~ImageSaverServer() {
    this->shared_memory_handler->close_connection();
}

/************************************************************************************************** */

void ImageSaverServer::reload_psf_data() {
    this->configs_translator.translate_psf();
    this->image_processor.reload();
}

void ImageSaverServer::setup_shared_memory_handler() {
    auto smh_config = this->config["smh_config"].as<std::string>();
    this->shared_memory_handler = std::make_unique<SharedMemoryHandler>(smh_config);
    this->shared_memory_handler->open_connection();
    this->shared_memory_handler->register_as_image_saver();
}

void ImageSaverServer::setup_zmq_client() {
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

    set_fgc_roi(
        config_fgc["roi_w"].as<int>(),
        config_fgc["roi_h"].as<int>(),
        fgc_timeout_ms,
        config_fgc["roi_x"].as<int>(),
        config_fgc["roi_y"].as<int>(),
        config_fgc["vbin"].as<int>(),
        config_fgc["hbin"].as<int>()
    );
}

void ImageSaverServer::set_fgc_roi(int roi_w, int roi_h, int timeout_ms, int roi_x, int roi_y, int vbin, int hbin) {
    fgc->setup(
        roi_w,
        roi_h,
        timeout_ms,
        roi_x,
        roi_y,
        vbin,
        hbin
    );
    this->roi_w = roi_w;
    this->roi_h = roi_h;
}

void ImageSaverServer::setup_image_capturer_thread() {
    flag_thread_running.store(false);
    this->image_capturer_thread = std::thread(&ImageSaverServer::capture_images, this);
}

void ImageSaverServer::setup_saver_worker() {
    this->image_saver_thread = std::thread(&ImageSaverServer::save_images, this);
}

void ImageSaverServer::capture_images() {
    std::vector<double_t> fls_counts;
    std::vector<uint8_t> occupancy;

    fls_counts.reserve(MAX_TRAP_ARRAY_SIZE);
    occupancy.reserve(MAX_TRAP_ARRAY_SIZE);

    while (!this->flag_thread_killed.load()) {
        if (flag_thread_running.load()) {
            std::vector<uint16_t> current_image = fgc->acquire_single_image();
            if (current_image.size() == 0) {
                flag_thread_running.store(false);
            } else {
                // Image processing
                auto trap_count = this->image_processor.get_trap_count();
                fls_counts.resize(trap_count);
                occupancy.resize(trap_count);
                this->image_processor.process(
                    roi_w,
                    image_counter,
                    current_image,
                    fls_counts,
                    occupancy
                );

                // Adding the processed data to the shared memory
                this->shared_memory_handler->save_trap_array_information(
                    trap_count,
                    fls_counts,
                    occupancy
                );

                // Signalling the master that a new image has been added
                this->shared_memory_handler->signal_done();

                // Add to save queue
                std::lock_guard<std::mutex> lock(cache_mutex);
                
                std::ostringstream filename_stream;
                filename_stream << "image-" << timestamp << "-" << image_counter << ".png";
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

void ImageSaverServer::save_images() {
    ImageBatch result;
    while (!this->flag_thread_killed.load()) {
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (images_cache.size() == 0) {
                continue;
            }
            result = images_cache.at(0);
            images_cache.erase(images_cache.begin());
        }

        ImageSaverServer::create_directory(boost::filesystem::path(std::get<1>(result)).parent_path());
        if (std::get<0>(result).size() == 0) { // Empty vector is a flag that the shot is finished, and we save a .done file to denote that.
            std::ofstream file(std::get<1>(result));
            file.close();
            continue;
        }
        cv::Mat image(roi_h, roi_w, CV_16U, std::get<0>(result).data());
        cv::imwrite(std::get<1>(result), image);
    }
}