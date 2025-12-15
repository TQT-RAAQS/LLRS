#ifndef RAMSEY_STABILIZER_H_
#define RAMSEY_STABILIZER_H_

#include "shared-memory-handler.h"
#include "llrs-lib/PreProc.h"
#include "labscript-address-utils.h"
#include "configs-translator.h"
#include "ramsey-stabilizer-metadata-saver.h"
#include "ramsey-stabilizer-labscript-config.h"
#include "fourier-analyzer.h"
#include "pid-loop-controller.h"
#include "microwave-awg-handler.h"
#include <unordered_map>
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

using namespace MicrowaveHandler;

class RamseyStabilizer {

    std::unique_ptr<RamseyStabilizerLabscriptConfig> labscript_config;

    YAML::Node configs;

    void setup_awg_handler();
    void prepare_awg();
    std::unique_ptr<MicrowaveAwgHandler> awg_handler;
    
    void reset_waveform_data();
    std::unordered_map<std::string, double> waveform_data;
    
    void setup_fourier_analyzer();
    std::unique_ptr<FourierAnalyzer> fourier_analyzer = nullptr;
    
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
    double phi, target_phi;
    double delta;
    void reset_pid();

    static std::string substitute_variables_in_signal(std::string s, const std::unordered_map<std::string, double>& vars);
    static std::vector<std::string> split_signal(const std::string& s, char delim);
public:
    RamseyStabilizer(const std::string config);
    ~RamseyStabilizer();

    void start();
    void stop();
};

#endif