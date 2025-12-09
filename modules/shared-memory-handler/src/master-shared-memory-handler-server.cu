#include "master-shared-memory-handler-server.h"

MasterSharedMemoryHandlerServer::MasterSharedMemoryHandlerServer(std::string config_name) {
    INFO << "Initializing MasterSharedMemoryHandlerServer with config: " << config_name << "\n";
    this->read_configs(config_name);
    this->setup_handler();
    this->setup_zmq_client();

    this->stop_flag.store(true);
    INFO << "MasterSharedMemoryHandlerServer initialized successfully" << "\n";
}

MasterSharedMemoryHandlerServer::~MasterSharedMemoryHandlerServer() {
    INFO << "Destroying MasterSharedMemoryHandlerServer";
    this->stop();
    INFO << "MasterSharedMemoryHandlerServer destroyed";
}

void MasterSharedMemoryHandlerServer::start() {
    if (this->server_thread.joinable()) {
        ERROR << "Cannot start server; server thread is already running" << "\n";
        throw std::runtime_error("Cannot start the server; the server seems to be already running.");
    }

    this->stop_flag.store(false);
    
    INFO << "Starting server thread" << "\n";
    this->server_thread = std::thread(&MasterSharedMemoryHandlerServer::server_worker, this);

    INFO << "Starting memory manager thread" << "\n";
    this->memory_manager_thread = std::thread(&MasterSharedMemoryHandlerServer::memory_manager_worker, this);
}

void MasterSharedMemoryHandlerServer::stop() {
    INFO << "Stopping server" << "\n";
    if (this->server_thread.joinable()) {
        this->stop_flag.store(true);
        this->server_thread.join();
    }
    INFO << "Server thread joined" << "\n";

    INFO << "Stopping memory manager" << "\n";
    if (this->memory_manager_thread.joinable()) {
        this->stop_flag.store(true);
        this->handler->clear_master_wait_semaphores();
        this->memory_manager_thread.join();
    }
    INFO << "Memory manager joind" << "\n";

    INFO << "Closing handler connection" << "\n";
    this->handler->close_connection();
    INFO << "Handler connection closed" << "\n";
}

void MasterSharedMemoryHandlerServer::wait_until_server_closed() {
    if (this->server_thread.joinable()) {
        INFO << "Waiting for server thread to close" << "\n";
        this->server_thread.join();
        INFO << "Server thread closed" << "\n";
    }
}

void MasterSharedMemoryHandlerServer::server_worker() {
    INFO << "Server worker started" << "\n";
    std::string request, response;
    while (!this->stop_flag.load()) {
        auto listen_output = this->zmq_client->listen(request);

        if (listen_output == 1) { // timed out
            continue; // just retry listening
        }

        if (listen_output) {
            ERROR << "Unexpected ZMQ error during listen" << "\n";
            break;
        }

        try {
            INFO << "Handling request: " << request << "\n";
            if (request == "exit") {
                INFO << "Received exit request. Sending 200 and stopping server." << "\n";
                this->zmq_client->send("200");
                break;
            }

            response = this->handle_request(request);
            INFO << "Sending response: " << response << "\n";
            this->zmq_client->send(response);
        } catch (const std::exception &e) {
            ERROR << "Exception when handling request: " << request << "; " << e.what() << "\n";
            this->zmq_client->send("500");
        }
    }
    INFO << "Server worker exiting" << "\n";
}

void MasterSharedMemoryHandlerServer::memory_manager_worker() {
    uint16_t current_image_count = this->handler->get_image_count();
    if (current_image_count != 0) {
        throw std::runtime_error("The initial image count is not 0. This is not expected.");
    }
    int16_t images_processed_count = SHOT_NOT_BEGUN_YET; // Flag that no shot has run yet.

    while (!this->stop_flag.load()) {
        this->handler->wait_for_image_saver(); // Wait until image saver sends a trigger.
        if (this->stop_flag.load()) break;

        current_image_count = this->handler->get_image_count();
        INFO << "Current image count is: " << std::to_string(current_image_count) << std::endl;
        INFO << "Number of processed images is: " << std::to_string(images_processed_count) << std::endl;
        if (current_image_count == 0 && images_processed_count == SHOT_NOT_BEGUN_YET) { // Transition to buffer; the shot has begun.
            this->handler->signal_processes(); // Tell the processes to initialize.
            
            this->handler->wait_for_processes(); // Wait for them to be initialized.
            if (this->stop_flag.load()) break;
            
            this->handler->signal_image_saver(); // Tell the image saver that the processes are ready and that it can proceed.

            images_processed_count = 0;
        
        }  else if (current_image_count > 0 && current_image_count > images_processed_count && images_processed_count != SHOT_NOT_BEGUN_YET) { // New image from the shot has arrived
            this->handler->signal_processes(); // Inform the processes that a new image has been taken.

            images_processed_count++; // The signal has been sent for one new image.

        } else if (current_image_count == images_processed_count) { // The shot is done.
            this->handler->signal_processes(); // Inform the processes that the experiment is over.
            
            this->handler->wait_for_processes(); // Wait for the processes to acknowledge completion.
            if (this->stop_flag.load()) break;

            this->handler->signal_image_saver(); // Signal the image saver that all processes are done.

            images_processed_count = SHOT_NOT_BEGUN_YET; // Flag that the current shot is over.

        } else {
            throw std::runtime_error("Unexpected case in the memory manager of the master shared memory handler. This is most likely a bug. Current image count: " + \
                std::to_string(current_image_count) + ", processed image count: " + std::to_string(images_processed_count) + ".");
        }
    }
}

std::string MasterSharedMemoryHandlerServer::handle_request(std::string request) {
    if (request == "hello") {
        INFO << "Received hello request" << "\n";
        return "hello";
    }
    else if (request == "get_image_count") {
        auto image_count = this->handler->get_image_count();
        INFO << "Responding with image count: " << image_count << "\n";
        return "200" + SERVER_DELIMITER + std::to_string(image_count);
    } 
    else if (request == "get_subscriber_pids") {
        auto subscriber_pids = this->handler->get_all_subscribers();
        if (subscriber_pids.empty()) {
            INFO << "No subscribers found" << "\n";
            return "204";
        }

        std::string response = "200";
        for (auto pid : subscriber_pids) {
            response += SERVER_DELIMITER + std::to_string(pid);
        }
        INFO << "Responding with subscriber PIDs: " << response << "\n";
        return response;
    } 
    else {
        ERROR << "Unknown request: " << request << "\n";
        return "404";
    }
}

void MasterSharedMemoryHandlerServer::read_configs(std::string config_name) {
    INFO << "Loading server configs from: " << config_name << "\n";
    this->configs = YAML::LoadFile(MASTER_SHARED_MEMORY_HANDLER_SERVER(config_name));
    INFO << "Server configs loaded successfully" << "\n";
}

void MasterSharedMemoryHandlerServer::setup_handler() {
    auto handler_config_name = this->configs["handler_config"].as<std::string>();
    INFO << "Setting up MasterSharedMemoryHandler with config: " << handler_config_name << "\n";
    this->handler = std::make_unique<MasterSharedMemoryHandler>(handler_config_name);
    this->handler->open_connection();
    INFO << "Handler connection opened" << "\n";
}

void MasterSharedMemoryHandlerServer::setup_zmq_client() {
    auto port = this->configs["port"].as<int>();
    auto timeout_ms = this->configs["listen_timeout_ms"].as<int>();
    INFO << "Setting up ZMQ client on port " << port << " with timeout " << timeout_ms << " ms" << "\n";
    this->zmq_client = std::make_unique<Server>(port, timeout_ms);
}