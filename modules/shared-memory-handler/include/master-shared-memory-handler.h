#ifndef MASTER_SHARED_MEMORY_HANDLER_H_
#define MASTER_SHARED_MEMORY_HANDLER_H_

#include "shared-memory-handler.h"
#include <csignal>

class MasterSharedMemoryHandler : public SharedMemoryHandler {

    void kill_subscribers();

public:
    MasterSharedMemoryHandler(std::string config_file_name) : SharedMemoryHandler(config_file_name) {}

    int open_connection() override;
};

#endif