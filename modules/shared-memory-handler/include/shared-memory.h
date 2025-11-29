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
#include <tuple>
#include <omp.h>

#define MAX_SUBSCRIPTION_COUNT 20
#define MAX_IMAGE_COUNT 200
#define MAX_ARRAY_WIDTH 100
#define MAX_ARRAY_HEIGHT 100
#define SHOT_NAME_MAX_SIZE 1000
#define PID_EMPTY 0

class MasterSharedMemoryHandler;

class alignas(alignof(pthread_mutex_t)) SharedMemory {
    pthread_mutex_t mtx;

    size_t shot_name_length;
    char shot_name[SHOT_NAME_MAX_SIZE];
    size_t subscription_count;
    size_t image_count;
    pid_t pid_image_saver = PID_EMPTY;

    std::array<bool, MAX_SUBSCRIPTION_COUNT> subscriber_finished_flags;
    std::array<pid_t, MAX_SUBSCRIPTION_COUNT> subscriber_pids;
    std::array<size_t, MAX_IMAGE_COUNT> trap_array_widths;
    std::array<size_t, MAX_IMAGE_COUNT> trap_array_heights;
    std::array<std::array<double_t, MAX_ARRAY_WIDTH*MAX_ARRAY_HEIGHT>, MAX_IMAGE_COUNT> traps_fluorescence_count;
    std::array<std::array<bool, MAX_ARRAY_WIDTH*MAX_ARRAY_HEIGHT>, MAX_IMAGE_COUNT> traps_occupancy;

    void initialize_mutex();
    void initialize_buffer();

    void mtx_lock();
    void mtx_unlock();

    std::tuple<int, std::vector<pid_t>> get_all_subscribers();
    std::tuple<int, std::vector<bool>> get_all_subscriber_finished_flags();

    friend class MasterSharedMemoryHandler;

public:
    ~SharedMemory();

    void initialize();
    bool register_image_saver(pid_t pid);
    
    bool save_trap_array_information(pid_t pid, 
                                     int trap_width, 
                                     int trap_height, 
                                     std::vector<double_t>& trap_fluorescence, 
                                     std::vector<uint8_t>& trap_occupied);
    size_t get_image_count();
    size_t get_trap_width(size_t  image_index);
    size_t get_trap_height(size_t  image_index);
    std::vector<double_t> get_trap_fluorescence(size_t image_index);
    std::vector<uint8_t> get_trap_occupancy(size_t image_index);

    size_t get_subscription_count();
    size_t add_subscriber(pid_t pid);
    size_t delete_subscriber(pid_t pid);

    bool are_subscribers_done();
    bool set_subscriber_finished_flag(pid_t pid, bool flag);
};

#endif