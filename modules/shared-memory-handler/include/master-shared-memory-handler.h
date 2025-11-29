#ifndef MASTER_SHARED_MEMORY_HANDLER_H_
#define MASTER_SHARED_MEMORY_HANDLER_H_

#include "shared-memory-handler.h"
#include <csignal>

class MasterSharedMemoryHandler : public SharedMemoryHandler {

    bool flag_kill_subscribers;
    int cleanup_pause_ms;
    std::thread cleanup_thread;
    
    void clear_broken_subscribers();
    void setup_cleanup_thread();
    void kill_subscribers();
    int clear_memory(); // returns 0 if successful, and -1 if unsuccessful. The -1 code could also be returned if the shared memory space has been already cleared.
    // NOTE: This function is automatically called by close connection if no processes are subscribed to the shared memory environment. However, you can call it directly as well. Be aware that
    // if any process is using this environment and you clear the memory, the other process will fail. Therefore, this function should not be called at all directly unless
    // we are plagued by zombie processes (processes that could not close connection before the process was terminated.)

public:
    MasterSharedMemoryHandler(std::string config_file_name);
    ~MasterSharedMemoryHandler();

    // Deleting copy constructors explicitly; cannot copy this object because of unmovable objcets like threads.
    MasterSharedMemoryHandler(const MasterSharedMemoryHandler&) = delete;
    MasterSharedMemoryHandler& operator=(const MasterSharedMemoryHandler&) = delete;

    void open_connection() override;
    void close_connection() override;

    std::vector<pid_t> get_all_subscribers();
    std::vector<bool> get_all_subscriber_finished_flags();
};

#endif