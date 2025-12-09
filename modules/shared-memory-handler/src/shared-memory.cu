#include "shared-memory-handler.h"

void SharedMemory::initialize(pid_t pid) {
    this->initialize_mutex();
    this->initialize_buffer();
    this->initialize_semaphores();

    this->register_master(pid);
    this->add_subscriber(pid);
}

void SharedMemory::register_master(pid_t pid) {
    this->pid_master = pid;
}

void SharedMemory::initialize_buffer() {
    this->mtx_lock();

    this->shot_address_length = 0;
    this->subscription_count = 0;
    this->image_count = 0;

    subscriber_pids.fill(PID_EMPTY);
    trap_array_sizes.fill(0);

    for (auto &arr : traps_fluorescence_count) {
        arr.fill(0);
    }
    
    for (auto &arr : traps_occupancy) {
        arr.fill(0);
    }

    this->mtx_unlock();
}

void SharedMemory::initialize_semaphores() {
    sem_init(&this->sem_master_wait_image_saver, 1, 0);
    sem_init(&this->sem_image_saver_wait_master, 1, 0);

    for (size_t i = 0; i < MAX_SUBSCRIPTION_COUNT; ++i) {
        sem_init(&this->sem_worker_wait_master[i], 1, 0);
        sem_init(&this->sem_master_wait_worker[i], 1, 0);
    }
}

void SharedMemory::destroy_semaphores() {
    sem_destroy(&this->sem_master_wait_image_saver);
    sem_destroy(&this->sem_image_saver_wait_master);

    for (size_t i = 0; i < MAX_SUBSCRIPTION_COUNT; ++i) {
        sem_destroy(&this->sem_worker_wait_master[i]);
        sem_destroy(&this->sem_master_wait_worker[i]);
    }
}


void SharedMemory::master_signal_image_saver(pid_t pid) {
    if (!this->is_master(pid)) { // Intentionally not locked
        throw std::runtime_error("This is not the master process, and is not allowed to call this function.");
    }

    this->mtx_lock();
    auto flag_image_saver_exists = this->pid_image_saver != PID_EMPTY;
    this->mtx_unlock();

    if (!flag_image_saver_exists) {
        throw std::runtime_error("The image saver is not registered. This is not expected.");
    }

    sem_post(&this->sem_image_saver_wait_master);
}

void SharedMemory::master_wait_for_image_saver(pid_t pid) {
    if (!this->is_master(pid)) { // Intentionally not locked
        throw std::runtime_error("This is not the master process, and is not allowed to call this function.");
    }

    sem_wait(&this->sem_master_wait_image_saver);
}

void SharedMemory::master_signal_others(pid_t pid) {
    if (!this->is_master(pid)) { // Intentionally not locked
        throw std::runtime_error("This is not the master process, and is not allowed to call this function.");
    }

    // No mutex intentionally
    auto& pis = this->pid_image_saver;
    auto& pm = this->pid_master;
    auto& sc = this->subscription_count;
    
    for (size_t i = 0; i < sc; ++i) {
        if (this->subscriber_pids[i] != pis && this->subscriber_pids[i] != pm) {
            sem_post(&this->sem_worker_wait_master[i]); // signal each worker individually
        }
    }
}

void SharedMemory::master_wait_for_others(pid_t pid) {
    if (!this->is_master(pid)) { // Intentionally not locked
        throw std::runtime_error("This is not the master process, and is not allowed to call this function.");
    }

    // No mutex intentionally
    auto& pis = this->pid_image_saver;
    auto& pm = this->pid_master;
    auto& sc = this->subscription_count;
    
    for (size_t i = 0; i < sc; ++i) {
        if (this->subscriber_pids[i] != pis && this->subscriber_pids[i] != pm) {
            sem_wait(&this->sem_master_wait_worker[i]); // signal each worker individually
        }
    }
}

void SharedMemory::submit_done_signal(pid_t pid) {
    int worker_index = this->is_valid_regular_process_pid(pid);

    if (worker_index != -1) { // regular worker
        sem_post(&this->sem_master_wait_worker[worker_index]);
    } else if (this->is_image_saver(pid)) {
        sem_post(&this->sem_master_wait_image_saver);
    } else {
        throw std::runtime_error("The provided PID is not a valid subscribed shared memory. This operation is invalid.");
    }
}


int SharedMemory::submit_wait(pid_t pid, uint8_t timeout_s) {
    int worker_index = this->is_valid_regular_process_pid(pid);

    if (worker_index != -1) { // regular worker
        if (timeout_s == 0) {
            return sem_wait(&this->sem_worker_wait_master[worker_index]);
        } else {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += timeout_s;

            return sem_timedwait(&this->sem_worker_wait_master[worker_index], &ts);
        }
    } else if (this->is_image_saver(pid)) {
        if (timeout_s == 0) {
            return sem_wait(&this->sem_image_saver_wait_master);
        } else {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += timeout_s;

            return sem_timedwait(&this->sem_image_saver_wait_master, &ts);
        }
    } else {
        throw std::runtime_error("The provided PID is not the image saver for the shared memory. This operation is invalid.");
    }
}

int8_t SharedMemory::is_valid_regular_process_pid(pid_t pid) {
    if (pid == PID_EMPTY || this->is_master(pid) || this->is_image_saver(pid)) {
        return -1;
    }

    this->mtx_lock();
    for (size_t i = 0; i < this->subscription_count; ++i) {
        if (this->subscriber_pids.at(i) == pid) {
            this->mtx_unlock();
            return i;
        }
    }
    this->mtx_unlock();

    return -1;
}

bool SharedMemory::is_image_saver(pid_t pid) {
    if (pid == PID_EMPTY) {
        return false;
    }

    this->mtx_lock();
    auto flag = this->pid_image_saver == pid;
    this->mtx_unlock();

    return flag;
}

bool SharedMemory::is_master(pid_t pid) {
    if (pid == PID_EMPTY) {
        return false;
    }

    auto flag = this->pid_master == pid; // No mutex on purpose

    return flag;
}

void SharedMemory::reset_semaphore(sem_t* sem) {
    int sval;
    while (sem_getvalue(sem, &sval) == 0 && sval > 0) {
        sem_trywait(sem);
    }
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

size_t SharedMemory::add_subscriber(pid_t pid) {
    this->mtx_lock();
    
    if (this->subscription_count == MAX_SUBSCRIPTION_COUNT) {
        throw std::runtime_error("The number of subscriptions is already maximized; it cannot be further increased.");
    }
    auto output = ++this->subscription_count;

    this->subscriber_pids.at(output - 1) = pid;
    
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

bool SharedMemory::save_trap_array_information(pid_t pid, 
                                               size_t trap_array_size,
                                               const std::vector<double_t>& trap_fluorescence, 
                                               const std::vector<uint8_t>& traps_occupancy) {
    auto image_count = this->get_image_count();

    // NOTE: WE INTENTIONALLY AVOID LOCKING THE MUTEX.
    if (pid != this->pid_image_saver) {
        throw std::runtime_error("This user does not have the permission to save trap array information to the shared space!");
    }
    if (image_count == MAX_IMAGE_COUNT) {
        throw std::runtime_error("The maximum number of images reached.");
    }
    if (trap_array_size > MAX_TRAP_ARRAY_SIZE) {
        throw std::runtime_error("The provided trap array information is larger than the maximum allowed.");
    }

    auto& N = trap_array_size;
    this->trap_array_sizes[image_count] = trap_array_size;

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

    auto& N = this->trap_array_sizes.at(image_index);
    
    std::vector<double_t> trap_fluorescence(N);
    std::copy_n(this->traps_fluorescence_count.at(image_index).begin(), N, trap_fluorescence.begin());
    return trap_fluorescence;
}

std::vector<uint8_t> SharedMemory::get_trap_occupancy(size_t  image_index) {
    auto image_count = this->get_image_count();

    if (image_index >= image_count) {
        throw std::runtime_error("The image_index provided is higher than the number of images available in the shared meomry.");
    }

    auto& N = this->trap_array_sizes.at(image_index);
    
    std::vector<uint8_t> traps_occupancy(N);
    std::copy_n(this->traps_occupancy.at(image_index).begin(), N, traps_occupancy.begin());
    return traps_occupancy;
}

size_t SharedMemory::get_trap_array_size(size_t  image_index) {
    auto image_count = this->get_image_count();

    if (image_index >= image_count) {
        throw std::runtime_error("The image_index provided is higher than the number of images available in the shared meomry.");
    }

    return this->trap_array_sizes.at(image_index);
}

std::string SharedMemory::get_shot_address() {
    this->mtx_lock();
    std::string output(this->shot_address, this->shot_address_length);
    this->mtx_unlock();

    return output;
}

bool SharedMemory::change_shot_address(pid_t pid, std::string new_shot_address) {
    if (!this->is_image_saver(pid) && !this->is_master(pid)) {
        ERROR << "The pid " << std::to_string(pid) << " is not the image saver or the master, and thus cannot set the shot name.\n";
        return false;
    }
    if (new_shot_address.length() > SHOT_ADDRESS_MAX_SIZE) {
        ERROR << "The provided shot name is longer than the maximum shot name allowed: " << std::to_string(new_shot_address.length()) << " > " << std::to_string(SHOT_ADDRESS_MAX_SIZE) << "\n";
        return false;
    }

    this->mtx_lock();
    this->shot_address_length = new_shot_address.length();
    std::copy(new_shot_address.begin(), new_shot_address.end(), this->shot_address);
    this->image_count = 0;
    this->mtx_unlock();

    return true;
}

bool SharedMemory::reset_image_count(pid_t pid) {
    if (!this->is_image_saver(pid) && !this->is_master(pid)) {
        ERROR << "The pid " << std::to_string(pid) << " is not the image saver or the master, and thus cannot set the shot name.\n";
        return false;
    }

    this->mtx_lock();
    this->image_count = 0;
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