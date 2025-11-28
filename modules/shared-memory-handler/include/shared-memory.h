#ifndef SHARED_MEMORY_H_
#define SHARED_MEMORY_H_

#include <iostream>
#include <yaml-cpp/yaml.h>
#include "llrs-lib/PreProc.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#include <cstddef>

#define MAX_SUBSCRIPTION_COUNT 20
#define MAX_IMAGE_COUNT 10
#define MAX_ARRAY_WIDTH 100
#define MAX_ARRAY_HEIGHT 100
#define SHOT_NAME_MAX_SIZE 1000

class MasterSharedMemoryHandler;

class alignas(alignof(pthread_mutex_t)) SharedMemory {
    pthread_mutex_t mtx;

    size_t shot_name_length;
    char shot_name[SHOT_NAME_MAX_SIZE];
    size_t subscription_count;
    size_t image_count;
    size_t trap_width;
    size_t trap_height;

    std::array<bool, MAX_SUBSCRIPTION_COUNT> subscriber_finished;
    std::array<pid_t, MAX_SUBSCRIPTION_COUNT> subscriber_pids;
    std::array<std::array<uint16_t, MAX_ARRAY_WIDTH*MAX_ARRAY_HEIGHT>, MAX_IMAGE_COUNT> trap_fluorescence_count;
    std::array<std::array<bool, MAX_ARRAY_WIDTH*MAX_ARRAY_HEIGHT>, MAX_IMAGE_COUNT> trap_occupied;

    void initialize_mutex();
    void initialize_buffer(size_t image_width, size_t image_height);

    void mtx_lock();
    void mtx_unlock();

    friend class MasterSharedMemoryHandler;

public:
    void initialize(size_t image_width, size_t image_height);

    int get_subscription_count();
    int increase_subscription_count(pid_t pid);
    int decrease_subscription_count(pid_t pid);

    bool are_subscribers_done();
};

#endif