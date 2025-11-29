#include "master-shared-memory-handler.h"
#include <csignal>

MasterSharedMemoryHandler::MasterSharedMemoryHandler(std::string config_file_name) 
    : SharedMemoryHandler(config_file_name) 
{
    INFO << "Initializing MasterSharedMemoryHandler with config: " << config_file_name << "\n";
    this->flag_kill_subscribers = this->configs["kill_subscribers"].as<bool>();
    this->cleanup_pause_ms = this->configs["cleanup_pause_ms"].as<int>();
    INFO << "flag_kill_subscribers=" << flag_kill_subscribers 
         << ", cleanup_pause_ms=" << cleanup_pause_ms << "\n";
}

MasterSharedMemoryHandler::~MasterSharedMemoryHandler() {
    INFO << "Destroying MasterSharedMemoryHandler for PID " << pid << "\n";
    this->close_connection();
    INFO << "MasterSharedMemoryHandler destroyed for PID " << pid << "\n";
}

void MasterSharedMemoryHandler::open_connection() {
    INFO << "Opening master shared memory connection: " << shared_memory_name;

    // Try to open existing memory
    shm_fd = shm_open(shared_memory_name.c_str(), O_RDWR, 0666);
    bool exists = (shm_fd != -1);
    if (exists) {
        INFO << "Shared memory exists. Clearing existing memory..." << "\n";
        this->shared_memory_void = mmap(nullptr, sizeof(SharedMemory),
                                        PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        this->shared_memory = reinterpret_cast<SharedMemory*>(shared_memory_void);
        this->flag_is_connected = true;
        this->close_connection();
        INFO << "Existing shared memory cleared." << "\n";
    }

    // Create new shared memory
    shm_fd = shm_open(shared_memory_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
    if (shm_fd == -1) {
        ERROR << "Failed to create shared memory: " << shared_memory_name << "\n";
        throw std::runtime_error("Failed to create shared memory");
    }
    ftruncate(shm_fd, sizeof(SharedMemory));
    shared_memory_void = mmap(nullptr, sizeof(SharedMemory),
                              PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    shared_memory = reinterpret_cast<SharedMemory*>(shared_memory_void);

    shared_memory->initialize();
    shared_memory->add_subscriber(pid);
    flag_is_connected = true;
    INFO << "Shared memory created and subscriber added for PID " << pid << "\n";

    this->setup_cleanup_thread();
}

void MasterSharedMemoryHandler::setup_cleanup_thread() {
    INFO << "Starting cleanup thread for broken subscribers" << "\n";
    this->cleanup_thread = std::thread(&MasterSharedMemoryHandler::clear_broken_subscribers, this);
}

void MasterSharedMemoryHandler::close_connection() {
    if (!this->is_connected()) {
        INFO << "Close connection called, but already disconnected for PID " << pid << "\n";
        return;
    }

    INFO << "Closing master shared memory connection for PID " << pid << "\n";

    if (this->flag_kill_subscribers) {
        INFO << "Killing subscribers as flag_kill_subscribers is true" << "\n";
        this->kill_subscribers();
    }
    
    flag_is_connected = false;
    if (this->cleanup_thread.joinable()) {
        INFO << "Joining cleanup thread" << "\n";
        this->cleanup_thread.join();
        INFO << "Cleanup thread joined" << "\n";
    }

    munmap(shared_memory_void, sizeof(SharedMemory));
    close(shm_fd);
    INFO << "Shared memory unmapped and file descriptor closed" << "\n";

    int result = this->clear_memory();
    if (result == 0) {
        INFO << "Shared memory cleared successfully" << "\n";
    } else {
        ERROR << "Failed to clear shared memory: " << shared_memory_name << "\n";
    }
}

void MasterSharedMemoryHandler::kill_subscribers() {
    INFO << "Killing all other subscribers" << "\n";
    this->shared_memory->mtx_lock();
    
    for (int i = 0; i < shared_memory->subscription_count; ++i) {
        pid_t p = shared_memory->subscriber_pids[i];
        if (p != this->pid && kill(p, 0) == 0) { // only if alive
            INFO << "Killing subscriber PID " << p << "\n";
            kill(p, SIGKILL);
        }
    }
    
    this->shared_memory->mtx_unlock();
    INFO << "Subscribers killed" << "\n";
}

void MasterSharedMemoryHandler::clear_broken_subscribers() {
    INFO << "Cleanup thread running for broken subscribers" << "\n";
    while (this->is_connected()) {
        int count;
        std::vector<pid_t> pid_list;
        std::tie(count, pid_list) = this->shared_memory->get_all_subscribers();

        for (size_t i = 0; i < count; ++i) {
            pid_t p = pid_list.at(i);
            if (p != this->pid && kill(p, 0) != 0) {
                INFO << "Removing broken subscriber PID " << p << "\n";
                this->shared_memory->delete_subscriber(p);
                if (this->shared_memory->pid_image_saver == p) {
                    INFO << "Broken subscriber was image saver. Resetting pid_image_saver" << "\n";
                    this->shared_memory->pid_image_saver = PID_EMPTY;
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(this->cleanup_pause_ms));
    }
    INFO << "Cleanup thread exiting" << "\n";
}

std::vector<pid_t> MasterSharedMemoryHandler::get_all_subscribers() {
    int count;
    std::vector<pid_t> pid_list;
    std::tie(count, pid_list) = this->shared_memory->get_all_subscribers();
    INFO << "Retrieved " << count << " subscribers" << "\n";
    return pid_list;
}

std::vector<bool> MasterSharedMemoryHandler::get_all_subscriber_finished_flags() {
    int count;
    std::vector<bool> flags;
    std::tie(count, flags) = this->shared_memory->get_all_subscriber_finished_flags();
    INFO << "Retrieved finished flags for " << count << " subscribers" << "\n";
    return flags;
}

int MasterSharedMemoryHandler::clear_memory() {
    INFO << "Clearing shared memory: " << shared_memory_name << "\n";
    return shm_unlink(this->shared_memory_name.c_str());
}