#include "shared-memory-handler.h"

SharedMemoryHandler::SharedMemoryHandler(std::string config_file_name) {
    INFO << "Initializing SharedMemoryHandler with config: " << config_file_name << "\n";
    this->read_configs(config_file_name);
    this->pid = getpid();
    INFO << "SharedMemoryHandler initialized for PID: " << pid << "\n";
}

SharedMemoryHandler::~SharedMemoryHandler() {
    INFO << "Destroying SharedMemoryHandler for PID: " << pid << "\n";
    this->close_connection();
    INFO << "SharedMemoryHandler destroyed" << "\n";
}

void SharedMemoryHandler::read_configs(std::string config_file_name) {
    INFO << "Reading configs from: " << config_file_name << "\n";
    std::string config_address = SHARED_MEMORY_HANDLER(config_file_name);
    this->configs = YAML::LoadFile(config_address);

    this->shared_memory_name = this->configs["shared_memory_name"].as<std::string>();
    INFO << "Shared memory name loaded: " << shared_memory_name << "\n";
}

bool SharedMemoryHandler::is_connected() {
    return this->flag_is_connected;
}

bool SharedMemoryHandler::save_trap_array_information(size_t trap_array_size,
                                                      const std::vector<double_t>& trap_fluorescence, 
                                                      const std::vector<uint8_t>& trap_occupied) {
    INFO << "Saving trap array information: " 
         << "trap array size=" << trap_array_size
         << ", fluorescence size=" << trap_fluorescence.size()
         << ", occupancy size=" << trap_occupied.size() << "\n";
    return this->shared_memory->save_trap_array_information(this->pid, trap_array_size, trap_fluorescence, trap_occupied);
}

size_t SharedMemoryHandler::get_image_count() {
    size_t count = this->shared_memory->get_image_count();
    INFO << "Retrieved image count: " << count << "\n";
    return count;
}

void SharedMemoryHandler::register_as_image_saver() {
    INFO << "Registering PID " << pid << " as image saver" << "\n";
    auto success = this->shared_memory->register_image_saver(this->pid);
    if (!success) {
        ERROR << "Failed to register PID " << pid << " as image saver";
        throw std::runtime_error("Failed to register as an image saver. Possibly another process has already registered as the image saver.");
    }
    INFO << "PID " << pid << " successfully registered as image saver" << "\n";
}

std::string SharedMemoryHandler::get_shot_address() {
    return this->shared_memory->get_shot_address();
}

void SharedMemoryHandler::change_shot_address(std::string new_shot_address) {
    this->shared_memory->change_shot_address(this->pid, new_shot_address);
}

void SharedMemoryHandler::reset_image_count() {
    INFO << "Resetting the number of images in the shared memory to 0\n";
    this->shared_memory->reset_image_count(this->pid);
}

size_t SharedMemoryHandler::get_trap_array_size(size_t image_index) {
    size_t width = this->shared_memory->get_trap_array_size(image_index);
    INFO << "Trap array size for image " << image_index << ": " << width << "\n";
    return width;
}

std::vector<double_t> SharedMemoryHandler::get_trap_fluorescence(size_t image_index) {
    INFO << "Getting trap fluorescence for image " << image_index << "\n";
    return this->shared_memory->get_trap_fluorescence(image_index);
}

std::vector<uint8_t> SharedMemoryHandler::get_trap_occupancy(size_t image_index) {
    INFO << "Getting trap occupancy for image " << image_index << "\n";
    return this->shared_memory->get_trap_occupancy(image_index);
}

void SharedMemoryHandler::open_connection() {
    INFO << "Opening connection to shared memory: " << shared_memory_name << "\n";
    this->shm_fd = shm_open(shared_memory_name.c_str(), O_RDWR, 0666);
    if (shm_fd == -1) {
        ERROR << "Shared memory does not exist: " << shared_memory_name << "\n";
        throw std::runtime_error("Shared memory does not exist: " + shared_memory_name + 
                                 ". Please open the master shared memory handler.");
    }

    this->shared_memory_void = mmap(nullptr, sizeof(SharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    this->shared_memory = reinterpret_cast<SharedMemory*>(shared_memory_void);
    shared_memory->add_subscriber(pid);
    this->flag_is_connected = true;
    INFO << "Connection opened successfully for PID " << pid << "\n";
}

void SharedMemoryHandler::close_connection() {
    if (!this->is_connected()) {
        INFO << "Close connection called, but already disconnected for PID " << pid << "\n";
        return;
    }

    INFO << "Closing connection for PID " << pid << "\n";
    shared_memory->delete_subscriber(pid);

    munmap(shared_memory_void, sizeof(SharedMemory));
    close(shm_fd);
    flag_is_connected = false;
    INFO << "Connection closed successfully for PID " << pid << "\n";
}

void SharedMemoryHandler::signal_done() {
    this->shared_memory->submit_done_signal(this->pid);
}

void SharedMemoryHandler::wait_for_update() {
    this->shared_memory->submit_wait(this->pid);
}