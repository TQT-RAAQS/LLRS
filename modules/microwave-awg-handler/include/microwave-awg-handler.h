#ifndef _MICROWAVE_AWG_HANDLER_H_
#define _MICROWAVE_AWG_HANDLER_H_

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

namespace MicrowaveHandler {

    #define MW_START_SEGMENT_INDEX 0
    #define MW_END_SEGMENT_INDEX 1
    #define MW_START_STEP_INDEX 0
    #define MW_END_STEP_INDEX 1
    #define MW_INITIAL_STEP_INDEX 2 // Make sure this is larger than the start and end steps
    #define MW_INITIAL_SEGMENT_INDEX 2 // Make sure this is larger than the start and end segments
    #define MW_MAX_STEP_REPETITION 1048575

    class MicrowaveAwgHandler {

    public:

        struct IQMixerWaveform {
            MicrowaveWaveforms::Waveform waveform;
            double time;
            double t_initial_pause;
            double duration;
            int64_t hash; // store the combined hash
        
            // Constructor computes the hash once
            IQMixerWaveform(const MicrowaveWaveforms::Waveform& wf, double time, double t_initial_pause, double duration)
                : waveform(wf), time(time), t_initial_pause(t_initial_pause), duration(duration)
            {
                auto hash_combine = [](int64_t seed, int64_t value) {
                    return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
                };
        
                // Compute hash from waveform variant
                int64_t h = boost::apply_visitor(
                    [](auto&& w) { return w.hash(); }, waveform);
        
                // Combine with other fields
                if (boost::get<MicrowaveWaveforms::SquarePulse>(&wf)) { // Square pulse
                    h = hash_combine(h, std::hash<double>{}(time));
                    h = hash_combine(h, std::hash<double>{}(t_initial_pause));
                    h = hash_combine(h, std::hash<double>{}(duration));
                } else if (boost::get<MicrowaveWaveforms::Pause>(&wf)) { // Pause
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

        MicrowaveSynthesizer::MicrowaveWaveformSynthesizer synthesizer;
        std::tuple<std::vector<IQMixerWaveform>, std::vector<int>> breakdown_waveforms(const std::vector<MicrowaveWaveforms::Waveform>& waveforms);
        int next_step_to_load_index = MW_INITIAL_STEP_INDEX; // Step to be used next by the load_waveforms function
        int step_to_run_index = MW_END_STEP_INDEX; // The first step to be run for the next shot
        int upload_iqmixer_waveform(IQMixerWaveform);
        int increment_step_index(int index, int step_size);
        
    public:
        AWG awg;
        MicrowaveAwgHandler(const std::string& handler_config, const std::string& awg_config);
        ~MicrowaveAwgHandler();

        void open_connection();
        void close_connection();
        void reload(bool flag_translate = true);
        void start();
        void stop();

        void upload_waveforms(const std::vector<MicrowaveWaveforms::Waveform>& waveforms);

        bool is_connected() const;

        int get_awg_step() { return this->awg.get_current_step(); };

        void clear_memory();
    };

}

#endif