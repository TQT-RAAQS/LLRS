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
    double error_signal;
    double new_frequency;
    double frequency_moving_average;
    int64_t timestamp;

    ShotInformation(std::string shot_address,
                    double error_signal,
                    double new_frequency,
                    double frequency_moving_average,
                    int64_t timestamp) : 
                    shot_address(shot_address),
                    error_signal(error_signal),
                    new_frequency(new_frequency),
                    frequency_moving_average(frequency_moving_average),
                    timestamp(timestamp)
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
    void add_to_queue(std::string shot_address,
                      double error_signal,
                      double new_frequency,
                      double frequency_moving_average,
                      int64_t timestamp);

    RamseyStabilizerMetadataSaver(YAML::Node configs);
    ~RamseyStabilizerMetadataSaver();
};

#endif