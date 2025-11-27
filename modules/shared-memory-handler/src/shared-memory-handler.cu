#include "shared-memory-handler.h"

SharedMemoryHandler::SharedMemoryHandler(std::string config_file_name) {
    this->read_configs(config_file_name);
    this->pid = getpid();
}

SharedMemoryHandler::~SharedMemoryHandler() {
    if (this->flag_is_connected) {
        this->close_connection();
    }
}

void SharedMemoryHandler::read_configs(std::string config_file_name) {
    std::string config_address = SHARED_MEMORY_HANDLER(config_file_name);
    this->configs = YAML::LoadFile(config_address);

    this->shared_memory_name = this->configs["shared_memory_name"].as<std::string>();
}

bool SharedMemoryHandler::is_connected() {
    return this->flag_is_connected;
}

int SharedMemoryHandler::open_connection() {
    auto output = this->setup_shared_memory();
    this->flag_is_connected = true;
    this->shared_memory->increment_subscription_count(this->pid);
    return output;
}

int SharedMemoryHandler::setup_shared_memory() {
    // Try to create the shared memory space. The flags here make sure that if we fail to create the new environment
    // (either because the shared memory space is already created, or some other unknown reason), the output is -1.
    this->shm_fd = shm_open(this->shared_memory_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
    int created = (shm_fd != -1);
    
    // If the shared memory space failed to be created, it might already exist.
    if (!created) {
        shm_fd = shm_open(this->shared_memory_name.c_str(), O_RDWR, 0666);
        if (shm_fd == -1) { // If we still failed, then something has gone wrong.
            throw std::runtime_error(
                std::string("Could neither create nor access the shared memory with the label ") + 
                this->shared_memory_name);
        }
    }

    if (created) { // If we have created the environment, we should also set the size of the shared space environment.
        ftruncate(this->shm_fd, sizeof(SharedMemory)); // Fix the size of the shared memory space.
    }
    
    this->shared_memory_void = mmap(nullptr, sizeof(SharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, this->shm_fd, 0);
    this->shared_memory = reinterpret_cast<SharedMemory*>(this->shared_memory_void);
    
    if (created) { // If we have created the environment, initialize the memotyr.
        this->shared_memory->initialize(100, 100);
    }

    return created + 1;
}

int SharedMemoryHandler::close_connection() {
    auto new_subscription_count = this->shared_memory->decrease_subscription_count(this->pid);
    
    munmap(this->shared_memory_void, sizeof(SharedMemory));
    close(this->shm_fd);
    this->flag_is_connected = false;

    if (new_subscription_count > 0) {
        return 1;
    }
    this->clear_memory();
    return 2;
}

int SharedMemoryHandler::clear_memory() {
    return shm_unlink(this->shared_memory_name.c_str());
}
