#ifndef MICROWAVE_AWG_HANDLER_H_
#define MICROWAVE_AWG_HANDLER_H_

#include "awg.hpp"
#include "timed-hash-queue.h"
#include "microwave-waveform-synthesizer.h"
#include "microwave-waveforms.h"
#include "yaml-cpp/yaml.h"
#include "llrs-lib/PreProc.h"
#include <unordered_map>
#include <fstream>
#include <cmath>
#include <tuple>
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>
#include <memory>

namespace MicrowaveHandler {

    #define MW_SHORT_SEGMENT_INDEX 0
    #define MW_INITIAL_SEGMENT_INDEX 1 // Make sure this is larger than the short segment

    #define MW_START_STEP_INDEX 0
    #define MW_END_STEP_INDEX 1
    #define MW_INITIAL_STEP_INDEX 2 // Make sure this is larger than the start and end steps
    
    #define MW_MAX_STEP_REPETITION 1048575

    class MicrowaveAwgHandler {

    public:

        struct IQMixerWaveform {
            MicrowaveHandler::Waveform waveform;
            double time;
            double t_initial_pause;
            double duration;
            int64_t hash; // store the combined hash
        
            // Constructor computes the hash once
            IQMixerWaveform(const MicrowaveHandler::Waveform& wf, double time, double t_initial_pause, double duration)
                : waveform(wf), time(time), t_initial_pause(t_initial_pause), duration(duration)
            {
                auto hash_combine = [](int64_t seed, int64_t value) {
                    return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
                };
        
                // Compute hash from waveform variant
                int64_t h = boost::apply_visitor(
                    [](auto&& w) { return w.hash(); }, waveform);
        
                // Combine with other fields
                if (boost::get<MicrowaveHandler::SquarePulse>(&wf)) { // Square pulse
                    h = hash_combine(h, std::hash<double>{}(time));
                    h = hash_combine(h, std::hash<double>{}(t_initial_pause));
                    h = hash_combine(h, std::hash<double>{}(duration));
                    h = hash_combine(h, std::hash<double>{}(1));
                } else if (boost::get<MicrowaveHandler::Square60Pulse>(&wf)) { // Square pulse
                    h = hash_combine(h, std::hash<double>{}(time));
                    h = hash_combine(h, std::hash<double>{}(t_initial_pause));
                    h = hash_combine(h, std::hash<double>{}(duration));
                    h = hash_combine(h, std::hash<double>{}(2));
                } else if (boost::get<MicrowaveHandler::Pause>(&wf)) { // Pause
                    h = hash_combine(h, std::hash<double>{}(duration));
                } else {
                    throw std::runtime_error("Unsupported waveform type for hashing.");
                }
        
                hash = h;
            }
        };

        // AWG awg;
        std::unordered_map<int64_t, int> hash_segment_index_map;
        TimedHashQueue awg_segments_queue;

        int max_step_size, min_segment_size, max_segment_count, default_pause_segment_size, segment_size_steps;
        double digital_offset_time;

        MicrowaveHandler::MicrowaveWaveformSynthesizer synthesizer;
        std::tuple<std::vector<IQMixerWaveform>, std::vector<int>> breakdown_waveforms(const std::vector<MicrowaveHandler::Waveform>& waveforms);
        int next_step_to_load_index = MW_INITIAL_STEP_INDEX; // Step to be used next by the load_waveforms function
        int step_to_run_index = MW_END_STEP_INDEX; // The first step to be run for the next shot
        int upload_iqmixer_waveform(IQMixerWaveform, bool lock_awg = false);
        int increment_step_index(int index, int step_size);

        std::mutex awg_mtx;
        int timer_worker_wait_time_ms;
        std::atomic<int64_t> streaming_time;
        std::atomic<bool> flag_timer_worker_kill;
        std::atomic<bool> flag_timer_worker_active;
        std::unique_ptr<std::thread> timer_worker_thread;
        void setup_timer_worker();
        void timer_worker();
        void reset_streaming_time();

    public:
        AWG awg;
        MicrowaveAwgHandler(const std::string& handler_config);
        ~MicrowaveAwgHandler();

        void open_connection();
        void close_connection();
        void reload(bool flag_translate = true);
        void start();
        void stop();
        void force_trigger();

        void upload_waveforms(const std::vector<MicrowaveHandler::Waveform>& waveforms);

        bool is_connected();

        int get_awg_step() { std::lock_guard<std::mutex> lock(this->awg_mtx); return this->awg.get_current_step(); };

        int64_t get_streaming_time(bool stop_timer = true);
        void stop_timer();

        void clear_memory();
    };

}

#endif