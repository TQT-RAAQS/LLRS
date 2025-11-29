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
    void register_as_image_saver();
    void set_finished_flag(bool flag);

    bool register_image_saver();

    bool save_trap_array_information(
        int trap_width, 
        int trap_height, 
        std::vector<double_t>& trap_fluorescence, 
        std::vector<uint8_t>& trap_occupied);
    size_t get_image_count();
    size_t get_trap_width(size_t image_index);
    size_t get_trap_height(size_t  image_index);
    std::vector<double_t> get_trap_fluorescence(size_t  image_index);
    std::vector<uint8_t> get_trap_occupancy(size_t  image_index);
    
    virtual void open_connection(); // 1 means successfully accessed the shared memory, and 2 means it succeeded but it had to create the shared memory since it was not instantiated yet by any other processes.
    virtual void close_connection(); // 1 means successfully unmapped the pointer, and 2 means it succeeded in doing so and cleared the memory space, as it is the last subscriber to the shared memory.
};

#endif