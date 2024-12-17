#include "image-saver-server.h"

ImageSaverServer::ImageSaverServer(std::string config_str) {
    std::string config_address = IMAGE_SAVER_SERVER(config_str);
    config = YAML::LoadFile(config_address);

    port = config["port"].as<int>();
    listen_timeout = config["listen_timeout"].as<int>();
    
    server = std::make_unique<Server>(port, listen_timeout);
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

void ImageSaverServer::transition_to_buffered(std::string h5_address) {
    INFO << "Processing a new shot: " << h5_address << std::endl;
}

void ImageSaverServer::transition_to_static() {
    INFO << "Transitioning to static" << std::endl;
}

void  ImageSaverServer::stop_acquiring_images() {
    // flag_acquire_image = false;
}