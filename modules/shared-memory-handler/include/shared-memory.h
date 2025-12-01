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
#define MAX_TRAP_ARRAY_SIZE 10000
#define SHOT_ADDRESS_MAX_SIZE 1000
#define PID_EMPTY 0

class MasterSharedMemoryHandler;

class alignas(alignof(pthread_mutex_t)) SharedMemory {
    static void reset_semaphore(sem_t* sem);

    pthread_mutex_t mtx;

    size_t shot_address_length;
    char shot_address[SHOT_ADDRESS_MAX_SIZE];
    size_t subscription_count;
    size_t image_count;
    pid_t pid_master = PID_EMPTY;
    pid_t pid_image_saver = PID_EMPTY;

    std::array<pid_t, MAX_SUBSCRIPTION_COUNT> subscriber_pids;
    std::array<size_t, MAX_TRAP_ARRAY_SIZE> trap_array_sizes;
    std::array<std::array<double_t, MAX_TRAP_ARRAY_SIZE>, MAX_IMAGE_COUNT> traps_fluorescence_count;
    std::array<std::array<bool, MAX_TRAP_ARRAY_SIZE>, MAX_IMAGE_COUNT> traps_occupancy;

    sem_t sem_master_wait_image_saver;
    sem_t sem_image_saver_wait_master;

    std::array<sem_t, MAX_SUBSCRIPTION_COUNT> sem_worker_wait_master;
    std::array<sem_t, MAX_SUBSCRIPTION_COUNT> sem_master_wait_worker;

    int8_t is_valid_regular_process_pid(pid_t pid); // Returns -1 if not valid, and returns the index if valid.
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
    
    bool change_shot_address(pid_t pid, std::string new_shot_address);
    bool reset_image_count(pid_t pid);
    bool save_trap_array_information(pid_t pid, 
                                     size_t trap_array_size, 
                                     const std::vector<double_t>& trap_fluorescence, 
                                     const std::vector<uint8_t>& trap_occupied);
    size_t get_image_count();
    size_t get_trap_array_size(size_t image_index);
    std::string get_shot_address();
    std::vector<double_t> get_trap_fluorescence(size_t image_index);
    std::vector<uint8_t> get_trap_occupancy(size_t image_index);
    size_t get_subscription_count();
    

    size_t add_subscriber(pid_t pid);
    size_t delete_subscriber(pid_t pid);

    void submit_done_signal(pid_t pid);
    void submit_wait(pid_t pid);
};

#endif