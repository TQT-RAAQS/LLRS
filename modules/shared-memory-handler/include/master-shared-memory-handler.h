#ifndef MASTER_SHARED_MEMORY_HANDLER_H_
#define MASTER_SHARED_MEMORY_HANDLER_H_

#include "shared-memory-handler.h"
#include <csignal>

class MasterSharedMemoryHandler : public SharedMemoryHandler {

    void kill_subscribers();
    int clear_memory(); // returns 0 if successful, and -1 if unsuccessful. The -1 code could also be returned if the shared memory space has been already cleared.
    // NOTE: This function is automatically called by close connection if no processes are subscribed to the shared memory environment. However, you can call it directly as well. Be aware that
    // if any process is using this environment and you clear the memory, the other process will fail. Therefore, this function should not be called at all directly unless
    // we are plagued by zombie processes (processes that could not close connection before the process was terminated.)

public:
    MasterSharedMemoryHandler(std::string config_file_name) : SharedMemoryHandler(config_file_name) {}

    void open_connection() override;
    void close_connection() override;
};

#endif