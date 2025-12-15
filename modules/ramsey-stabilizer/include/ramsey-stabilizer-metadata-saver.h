#ifndef RAMSEY_STABILIZER_METADATA_SAVER_H_
#define RAMSEY_STABILIZER_METADATA_SAVER_H_
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <thread>
#include <tuple>
#include <atomic>
#include <deque>
#include <mutex>
#include "labscript-address-utils.h"

struct ShotInformation {

    std::string shot_address;
    double extracted_phase;
    double old_detuning;
    double new_detuning;

    ShotInformation(std::string shot_address,
                    double extracted_phase,
                    double old_detuning,
                    double new_detuning) : 
                    shot_address(shot_address),
                    extracted_phase(extracted_phase),
                    old_detuning(old_detuning),
                    new_detuning(new_detuning)
                    {}
    ShotInformation() = default;
};

class RamseyStabilizerMetadataSaver{

    YAML::Node configs;
    std::string image_folder;

    std::unique_ptr<std::thread> thread_worker = nullptr;
    std::atomic<bool> thread_killed;
    std::deque<ShotInformation> queue;
    void worker();
    void save_to_file(const ShotInformation& s);

    std::mutex mtx;

public:

    void start();
    void stop();
    void add_to_queue(std::string shot_address, double extracted_phase, double old_detuning, double new_detuning);

    RamseyStabilizerMetadataSaver(YAML::Node configs);
    ~RamseyStabilizerMetadataSaver();
};

#endif