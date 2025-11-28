#ifndef SHARED_MEMORY_HANDLER_H_
#define SHARED_MEMORY_HANDLER_H_

#include "shared-memory.h"

class SharedMemoryHandler {
    
    void read_configs(std::string config_file_name);
    
protected:
    YAML::Node configs;
    std::string shared_memory_name;
    int shm_fd;
    void* shared_memory_void = nullptr;
    bool flag_is_connected = false;

    SharedMemory* shared_memory = nullptr;
    pid_t pid;
    
public:
    SharedMemoryHandler(std::string config_file_name);
    virtual ~SharedMemoryHandler();

    bool is_connected();
    
    virtual void open_connection(); // 1 means successfully accessed the shared memory, and 2 means it succeeded but it had to create the shared memory since it was not instantiated yet by any other processes.
    virtual void close_connection(); // 1 means successfully unmapped the pointer, and 2 means it succeeded in doing so and cleared the memory space, as it is the last subscriber to the shared memory.
};

#endif