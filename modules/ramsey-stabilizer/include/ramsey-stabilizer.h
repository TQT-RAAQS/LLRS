#ifndef RAMSEY_STABILIZER_H_
#define RAMSEY_STABILIZER_H_

#include "shared-memory-handler.h"
#include "llrs-lib/PreProc.h"
#include "labscript-address-utils.h"
#include "configs-translator.h"
#include "ramsey-stabilizer-metadata-saver.h"
#include "ramsey-stabilizer-labscript-config.h"
#include "phase-extractor.h"
#include "pid-loop-controller.h"
#include <string>
#include <boost/filesystem.hpp>
#include <semaphore.h>
#include <thread>
#include <chrono>
#include <ctime>
#include <fstream>
#include <fftw3.h>
#include <omp.h>
#include <cmath>
#include <complex>

class RamseyStabilizer {

    std::unique_ptr<RamseyStabilizerLabscriptConfig> labscript_config;

    YAML::Node configs;
    
    void setup_fft();
    std::unique_ptr<PhaseExtractor> phase_extractor = nullptr;
    
    std::unique_ptr<SharedMemoryHandler> smh = nullptr;
    std::string last_shot_address = "";
    std::string last_experiment_folder = "";
    
    std::unique_ptr<std::thread> thread_worker = nullptr;
    std::atomic<bool> thread_worker_killed;
    void worker_function();

    bool flag_configs_translator;

    void setup_memory_handler();

    void transition_to_buffered();

    void process_image(int8_t image_index);
    std::vector<uint8_t> oc0, oc1; // Occupancy flags in images 0 and 1.

    std::unique_ptr<RamseyStabilizerMetadataSaver> saver;
    void setup_saver();

    std::unique_ptr<PIDLoopController> pid_controller;
    double phi;
    double delta;
    void reset_pid();
public:
    RamseyStabilizer(const std::string config);
    ~RamseyStabilizer();

    void start();
    void stop();
};

#endif