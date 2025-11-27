#ifndef SHARED_MEMORY_HANDLER_H_
#define SHARED_MEMORY_HANDLER_H_

#include "shared-memory.h"

class SharedMemoryHandler {
    
    YAML::Node configs;
    std::string shared_memory_name;
    
    void read_configs(std::string config_file_name);
    int setup_shared_memory();
    int shm_fd;
    void* shared_memory_void = nullptr;
    bool flag_is_connected = false;
    
protected:
    SharedMemory* shared_memory = nullptr;
    pid_t pid;
    int clear_memory(); // returns 0 if successful, and -1 if unsuccessful. The -1 code could also be returned if the shared memory space has been already cleared.
    // NOTE: This function is automatically called by close connection if no processes are subscribed to the shared memory environment. However, you can call it directly as well. Be aware that
    // if any process is using this environment and you clear the memory, the other process will fail. Therefore, this function should not be called at all directly unless
    // we are plagued by zombie processes (processes that could not close connection before the process was terminated.)
    
public:
    SharedMemoryHandler(std::string config_file_name);
    ~SharedMemoryHandler();

    bool is_connected();
    
    virtual int open_connection(); // 1 means successfully accessed the shared memory, and 2 means it succeeded but it had to create the shared memory since it was not instantiated yet by any other processes.
    int close_connection(); // 1 means successfully unmapped the pointer, and 2 means it succeeded in doing so and cleared the memory space, as it is the last subscriber to the shared memory.
};

#endif