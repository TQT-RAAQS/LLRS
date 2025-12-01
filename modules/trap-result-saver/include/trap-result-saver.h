#ifndef TRAP_RESULT_SAVER_H_
#define TRAP_RESULT_SAVER_H_

#include "shared-memory-handler.h"
#include "llrs-lib/PreProc.h"
#include "labscript-address-utils.h"
#include <string>

#define SHOT_NOT_BEGUN -1
#define IMAGE_FOLDER_NAME 

class TrapResultSaver {

    YAML::Node configs;

    std::unique_ptr<SharedMemoryHandler> memory_handler;
    std::atomic<bool> thread_killed;
    std::unique_ptr<std::thread> saver_thread;

    std::string image_folder_name;

    void setup_memory_handler();
    void saver_worker();

public:
    TrapResultSaver(const std::string config);
    ~TrapResultSaver();

    void start();
    void stop();
};

#endif