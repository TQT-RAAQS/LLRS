#ifndef MASTER_SHRAED_MEMORY_HANDLER_H_
#define MASTER_SHRAED_MEMORY_HANDLER_H_

#include "master-shared-memory-handler.h"
#include "server.hpp"

#define SERVER_DELIMITER std::string("|")

class MasterSharedMemoryHandlerServer {

    YAML::Node configs;
    std::unique_ptr<MasterSharedMemoryHandler> handler;
    std::unique_ptr<Server> zmq_client; // TODO: Server is a bad name; this is actually a zmq_client, and thus the variable name.
    std::atomic<bool> stop_flag;

    std::thread server_thread;
    std::thread memory_manager_thread;

    void read_configs(std::string config_name);
    void setup_handler();
    void setup_zmq_client();

    void server_worker();
    void memory_manager_worker();
    std::string handle_request(std::string request);

public:
    MasterSharedMemoryHandlerServer(std::string config_name);
    ~MasterSharedMemoryHandlerServer();

    void start();
    void stop();
    void wait_until_server_closed();
};

#endif