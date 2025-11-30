#ifndef SHARED_MEMORY_H_
#define SHARED_MEMORY_H_

#include <iostream>
#include <yaml-cpp/yaml.h>
#include "llrs-lib/PreProc.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
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
    static void reset_semaphore(sem_t* sem);

    pthread_mutex_t mtx;

    size_t shot_name_length;
    char shot_name[SHOT_NAME_MAX_SIZE];
    size_t subscription_count;
    size_t image_count;
    pid_t pid_master = PID_EMPTY;
    pid_t pid_image_saver = PID_EMPTY;

    std::array<pid_t, MAX_SUBSCRIPTION_COUNT> subscriber_pids;
    std::array<size_t, MAX_IMAGE_COUNT> trap_array_widths;
    std::array<size_t, MAX_IMAGE_COUNT> trap_array_heights;
    std::array<std::array<double_t, MAX_ARRAY_WIDTH*MAX_ARRAY_HEIGHT>, MAX_IMAGE_COUNT> traps_fluorescence_count;
    std::array<std::array<bool, MAX_ARRAY_WIDTH*MAX_ARRAY_HEIGHT>, MAX_IMAGE_COUNT> traps_occupancy;

    sem_t sem_master_wait_image_saver;
    sem_t sem_image_saver_wait_master;
    sem_t sem_others_wait_master;
    sem_t sem_master_wait_others;

    bool is_valid_regular_process_pid(pid_t pid);
    bool is_image_saver(pid_t pid);
    bool is_master(pid_t pid);

    void initialize(pid_t pid_master);
    void initialize_mutex();
    void initialize_buffer();
    void initialize_semaphores();

    void register_master(pid_t pid);

    void destroy_semaphores();

    void mtx_lock();
    void mtx_unlock();

    void master_signal_image_saver(pid_t pid);
    void master_wait_for_image_saver(pid_t pid);
    void master_signal_others(pid_t pid);
    void master_wait_for_others(pid_t pid);

    std::tuple<int, std::vector<pid_t>> get_all_subscribers();

    friend class MasterSharedMemoryHandler;

public:
    ~SharedMemory();

    bool register_image_saver(pid_t pid);
    
    bool change_shot(pid_t pid, std::string new_shot_name);
    bool save_trap_array_information(pid_t pid, 
                                     int trap_width, 
                                     int trap_height, 
                                     std::vector<double_t>& trap_fluorescence, 
                                     std::vector<uint8_t>& trap_occupied);
    size_t get_image_count();
    size_t get_trap_width(size_t  image_index);
    size_t get_trap_height(size_t  image_index);
    std::string get_shot_name();
    std::vector<double_t> get_trap_fluorescence(size_t image_index);
    std::vector<uint8_t> get_trap_occupancy(size_t image_index);
    size_t get_subscription_count();
    

    size_t add_subscriber(pid_t pid);
    size_t delete_subscriber(pid_t pid);

    void submit_done_signal(pid_t pid);
    void submit_wait(pid_t pid);
};

#endif