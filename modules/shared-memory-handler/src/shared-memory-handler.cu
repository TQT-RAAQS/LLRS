#include "shared-memory-handler.h"

SharedMemoryHandler::SharedMemoryHandler(std::string config_file_name) {
    this->read_configs(config_file_name);
    this->pid = getpid();
}

SharedMemoryHandler::~SharedMemoryHandler() {
    this->close_connection();
}

void SharedMemoryHandler::read_configs(std::string config_file_name) {
    std::string config_address = SHARED_MEMORY_HANDLER(config_file_name);
    this->configs = YAML::LoadFile(config_address);

    this->shared_memory_name = this->configs["shared_memory_name"].as<std::string>();
}

bool SharedMemoryHandler::is_connected() {
    return this->flag_is_connected;
}

void SharedMemoryHandler::open_connection() {
    this->shm_fd = shm_open(shared_memory_name.c_str(), O_RDWR, 0666);
    if (shm_fd == -1)
        throw std::runtime_error("Shared memory does not exist: " + shared_memory_name + 
        ". Please open the shared memory handler.");

    this->shared_memory_void = mmap(nullptr, sizeof(SharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    this->shared_memory = reinterpret_cast<SharedMemory*>(shared_memory_void);
    shared_memory->increase_subscription_count(pid);
    this->flag_is_connected = true;
}

void SharedMemoryHandler::close_connection() {
    if (!this->is_connected()) {
        return;
    }

    shared_memory->decrease_subscription_count(pid);

    munmap(shared_memory_void, sizeof(SharedMemory));
    close(shm_fd);
    flag_is_connected = false;
}
