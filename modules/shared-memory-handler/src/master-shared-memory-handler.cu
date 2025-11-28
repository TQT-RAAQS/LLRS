#include "master-shared-memory-handler.h"
#include <csignal>

void MasterSharedMemoryHandler::open_connection() {
    // Try to open existing memory
    shm_fd = shm_open(shared_memory_name.c_str(), O_RDWR, 0666);
    bool exists = (shm_fd != -1);

    if (exists) {
        this->shared_memory_void = mmap(nullptr, sizeof(SharedMemory),
                                  PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        this->shared_memory = reinterpret_cast<SharedMemory*>(shared_memory_void);
        this->flag_is_connected = true;
        this->close_connection();
    }

    // Create new shared memory
    shm_fd = shm_open(shared_memory_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
    if (shm_fd == -1)
        throw std::runtime_error("Failed to create shared memory");

    ftruncate(shm_fd, sizeof(SharedMemory));
    shared_memory_void = mmap(nullptr, sizeof(SharedMemory),
                              PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    shared_memory = reinterpret_cast<SharedMemory*>(shared_memory_void);

    shared_memory->initialize(100, 100);
    shared_memory->increase_subscription_count(pid);
    flag_is_connected = true;
}

void MasterSharedMemoryHandler::close_connection() {
    if (!this->is_connected()) {
        return;
    }

    if (this->flag_kill_subscribers) {
        this->kill_subscribers();
    }
    
    munmap(shared_memory_void, sizeof(SharedMemory));
    close(shm_fd);
    flag_is_connected = false;

    this->clear_memory();
}

void MasterSharedMemoryHandler::kill_subscribers() {
    this->shared_memory->mtx_lock();
    
    for (int i = 0; i < shared_memory->subscription_count; ++i) {
        pid_t p = shared_memory->subscriber_pids[i];
        if (p != pid && kill(p, 0) == 0) { // only if alive
            kill(p, SIGKILL);
        }
    }
    
    this->shared_memory->mtx_unlock();
}

int MasterSharedMemoryHandler::clear_memory() {
    return shm_unlink(this->shared_memory_name.c_str());
}

MasterSharedMemoryHandler::~MasterSharedMemoryHandler() {
    this->close_connection( );
}