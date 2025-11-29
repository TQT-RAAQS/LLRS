#include "shared-memory-handler.h"

void SharedMemory::initialize() {
    this->initialize_mutex();
    this->initialize_buffer();
}

void SharedMemory::initialize_buffer() {
    this->mtx_lock();

    this->shot_name_length = 0;
    this->subscription_count = 0;
    this->image_count = 0;

    subscriber_finished_flags.fill(false);
    subscriber_pids.fill(PID_EMPTY);
    trap_array_widths.fill(0);
    trap_array_heights.fill(0);

    for (auto &arr : traps_fluorescence_count) {
        arr.fill(0);
    }
    
    for (auto &arr : traps_occupancy) {
        arr.fill(0);
    }

    this->mtx_unlock();
}

size_t SharedMemory::get_subscription_count() {
    this->mtx_lock();
    auto output = this->subscription_count;
    this->mtx_unlock();
    return output;
}

std::tuple<int, std::vector<pid_t>> SharedMemory::get_all_subscribers() {
    this->mtx_lock();
    size_t count = this->subscription_count;
    std::vector<int> pid_list(count);
    for (size_t i = 0; i < count; ++i) {
        pid_list.at(i) = this->subscriber_pids.at(i);
    }

    this->mtx_unlock();

    return std::make_tuple(count, pid_list);
}

std::tuple<int, std::vector<bool>> SharedMemory::get_all_subscriber_finished_flags() {
    this->mtx_lock();
    size_t count = this->subscription_count;
    std::vector<bool> flags(count);
    for (size_t i = 0; i < count; ++i) {
        flags.at(i) = this->subscriber_finished_flags.at(i);
    }

    this->mtx_unlock();

    return std::make_tuple(count, flags);
}

size_t SharedMemory::add_subscriber(pid_t pid) {
    this->mtx_lock();
    
    if (this->subscription_count == MAX_SUBSCRIPTION_COUNT) {
        throw std::runtime_error("The number of subscriptions is already maximized; it cannot be further increased.");
    }
    auto output = ++this->subscription_count;

    this->subscriber_pids.at(output - 1) = pid;
    this->subscriber_finished_flags.at(output - 1) = false;
    
    this->mtx_unlock();
    
    return output;
}

size_t SharedMemory::delete_subscriber(pid_t pid) {
    this->mtx_lock();

    if (this->pid_image_saver == pid) {
        this->pid_image_saver = PID_EMPTY;
    }

    if (this->subscription_count == 0) {
        this->mtx_unlock();
        return 0;
    }
    
    // auto output = --this->subscription_count;
    auto output = this->subscription_count - 1;
    
    size_t i;
    for (i = 0; i < output; ++i) {
        if (this->subscriber_pids.at(i) == pid) {
            this->subscriber_pids.at(i) = this->subscriber_pids.at(output);
            this->subscriber_finished_flags.at(i) = this->subscriber_finished_flags.at(output);
            break;
        }
    }
    if (i == output) {
        output += (this->subscriber_pids.at(i) != pid);
    }

    this->subscription_count = output;

    this->mtx_unlock();
    
    return output;
}

bool SharedMemory::register_image_saver(pid_t pid) {
    this->mtx_lock();

    auto old_pid = this->pid_image_saver;
    bool pid_set = old_pid != PID_EMPTY;
    if (!pid_set) {
        this->pid_image_saver = pid;
    }

    this->mtx_unlock();

    return !pid_set;
}

bool SharedMemory::set_subscriber_finished_flag(pid_t pid, bool flag) {
    this->mtx_lock();
    
    size_t i;
    for (i = 0; i < this->subscription_count; ++i) {
        if (this->subscriber_pids.at(i) == pid) {
            break;
        }
    }

    bool found = i < this->subscription_count;
    if (found) {
        this->subscriber_finished_flags.at(i) = flag;
    }

    this->mtx_unlock();

    return found;
}

bool SharedMemory::save_trap_array_information(pid_t pid, 
                                               int trap_width,
                                               int trap_height,
                                               std::vector<double_t>& trap_fluorescence, 
                                               std::vector<uint8_t>& traps_occupancy) {
    auto image_count = this->get_image_count();

    // NOTE: WE INTENTIONALLY AVOID LOCKING THE MUTEX.
    if (pid != this->pid_image_saver) {
        throw std::runtime_error("This user does not have the permission to save trap array information to the shared space!");
    }
    if (image_count == MAX_IMAGE_COUNT) {
        throw std::runtime_error("The maximum number of images reached.");
    }
    if (trap_width > MAX_ARRAY_WIDTH || trap_height > MAX_ARRAY_HEIGHT) {
        throw std::runtime_error("The provided trap array information is larger than the maximum allowed.");
    }

    int N = trap_width * trap_height;
    this->trap_array_widths[image_count] = trap_width;
    this->trap_array_heights[image_count] = trap_height;

    std::copy(trap_fluorescence.begin(), trap_fluorescence.begin() + N, this->traps_fluorescence_count[image_count].begin());
    std::copy(traps_occupancy.begin(), traps_occupancy.begin() + N, this->traps_occupancy[image_count].begin());

    this->mtx_lock();
    this->image_count = image_count + 1;
    this->mtx_unlock();

    return true;
}

size_t SharedMemory::get_image_count() {
    this->mtx_lock();
    auto image_count = this->image_count;
    this->mtx_unlock();

    return image_count;
}

std::vector<double_t> SharedMemory::get_trap_fluorescence(size_t image_index) {
    auto image_count = this->get_image_count();

    if (image_index >= image_count) {
        throw std::runtime_error("The image_index provided is higher than the number of images available in the shared meomry.");
    }

    auto trap_width = this->trap_array_widths.at(image_index);
    auto trap_height = this->trap_array_heights.at(image_index);
    auto N = trap_width * trap_height;
    
    std::vector<double_t> trap_fluorescence(N);
    std::copy_n(this->traps_fluorescence_count.at(image_index).begin(), N, trap_fluorescence.begin());
    return trap_fluorescence;
}

std::vector<uint8_t> SharedMemory::get_trap_occupancy(size_t  image_index) {
    auto image_count = this->get_image_count();

    if (image_index >= image_count) {
        throw std::runtime_error("The image_index provided is higher than the number of images available in the shared meomry.");
    }

    auto trap_width = this->trap_array_widths.at(image_index);
    auto trap_height = this->trap_array_heights.at(image_index);
    auto N = trap_width * trap_height;
    
    std::vector<uint8_t> traps_occupancy(N);
    std::copy_n(this->traps_occupancy.at(image_index).begin(), N, traps_occupancy.begin());
    return traps_occupancy;
}

size_t SharedMemory::get_trap_width(size_t  image_index) {
    auto image_count = this->get_image_count();

    if (image_index >= image_count) {
        throw std::runtime_error("The image_index provided is higher than the number of images available in the shared meomry.");
    }

    return this->trap_array_widths.at(image_index);
}

size_t SharedMemory::get_trap_height(size_t  image_index) {
    auto image_count = this->get_image_count();

    if (image_index >= image_count) {
        throw std::runtime_error("The image_index provided is higher than the number of images available in the shared meomry.");
    }
    
    return this->trap_array_heights.at(image_index);
}

bool SharedMemory::are_subscribers_done() {
    this->mtx_lock();

    for (size_t i = 0; i < this->subscription_count; ++i) {
        if (!this->subscriber_finished_flags.at(i)) {
            this->mtx_unlock();
            return false;
        }
    }

    this->mtx_unlock();
    return true;
}

void SharedMemory::mtx_lock() {
    int rc = pthread_mutex_lock(&this->mtx);
    if (rc == EOWNERDEAD) {
        pthread_mutex_consistent(&this->mtx);
    }
}

void SharedMemory::mtx_unlock() {
    pthread_mutex_unlock(&this->mtx);
}

void SharedMemory::initialize_mutex() {
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
    pthread_mutex_init(&this->mtx, &attr);
}

SharedMemory::~SharedMemory() {
    pthread_mutex_destroy(&this->mtx);
}