#ifndef TRAP_RESULT_SAVER_H_
#define TRAP_RESULT_SAVER_H_

#include "shared-memory-handler.h"
#include "llrs-lib/PreProc.h"
#include "labscript-address-utils.h"
#include <string>
#include <boost/filesystem.hpp>
#include <semaphore.h>
#include <thread>
#include <chrono>
#include <ctime>
#include <fstream>

#define IMAGE_FOLDER_NAME 

using ImageTrapResult = std::tuple<std::vector<double_t>, std::vector<uint8_t>>; // fluorescence and occupancy, each of size trap_size
using ShotTrapResult = std::tuple< std::string, std::vector<ImageTrapResult> >; // Address + vector of size image_count

class TrapResultSaver {

    YAML::Node configs;
    sem_t* saving_semaphore;

    std::vector< ShotTrapResult > trap_results;

    std::unique_ptr<SharedMemoryHandler> memory_handler;
    std::atomic<bool> thread_killed;
    std::unique_ptr<std::thread> data_retriever_thread;
    std::unique_ptr<std::thread> saver_thread;

    std::string image_folder_name;

    void setup_memory_handler();
    void retriever_worker();
    void saver_worker();
    void add_data_to_queue(size_t image_index, std::string save_directory);
    void setup_semaphore();
    void save_to_file(ShotTrapResult& shot_trap_result);

public:
    TrapResultSaver(const std::string config);
    ~TrapResultSaver();

    void start();
    void stop();
};

#endif