#include "shared-memory-handler.h"

void SharedMemory::initialize(size_t trap_width, size_t trap_height) {
    this->initialize_mutex();
    this->initialize_buffer(trap_width, trap_height);
}

void SharedMemory::initialize_buffer(size_t trap_width, size_t trap_height) {
    this->mtx_lock();

    this->shot_name_length = 0;
    this->subscription_count = 0;
    this->image_count = 0;
    this->trap_width = trap_width;
    this->trap_height = trap_height;

    for (size_t i = 0; i < MAX_SUBSCRIPTION_COUNT; ++i) {
        subscriber_finished.at(i) = false;
        subscriber_pids.at(i) = 0;
    }
    for (size_t i = 0; i < MAX_IMAGE_COUNT; ++i) {
        for (size_t j = 0; j < trap_width * trap_height; j++) {
            this->trap_fluorescence_count.at(i).at(j) = 0;
            this->trap_occupied.at(i).at(j) = 0;
        }
    }

    this->mtx_unlock();
}

int SharedMemory::get_subscription_count() {
    this->mtx_lock();
    auto output = this->subscription_count;
    this->mtx_unlock();
    return output;
}

int SharedMemory::increase_subscription_count(pid_t pid) {
    this->mtx_lock();
    
    if (this->subscription_count == MAX_SUBSCRIPTION_COUNT) {
        throw std::runtime_error("The number of subscriptions is already maximized; it cannot be further increased.");
    }
    auto output = ++this->subscription_count;

    this->subscriber_pids.at(output - 1) = pid;
    this->subscriber_finished.at(output - 1) = false;
    
    this->mtx_unlock();
    
    return output;
}

int SharedMemory::decrease_subscription_count(pid_t pid) {
    this->mtx_lock();

    if (this->subscription_count == 0) {
        throw std::runtime_error("The number of subscriptions is 0; it cannot be further decremented.");
    }
    auto output = --this->subscription_count;
    
    for (size_t i = 0; i < output; ++i) {
        if (this->subscriber_pids.at(i) == pid) {
            this->subscriber_pids.at(i) = this->subscriber_pids.at(output);
            this->subscriber_finished.at(i) = this->subscriber_finished.at(output);
            break;
        }
    }

    this->mtx_unlock();
    
    return output;
}

bool SharedMemory::are_subscribers_done() {
    this->mtx_lock();

    for (size_t i = 0; i < this->subscription_count; ++i) {
        if (!this->subscriber_finished.at(i)) {
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