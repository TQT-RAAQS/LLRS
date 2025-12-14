#ifndef _MICROWAVE_AWG_HANDLER_H_
#define _MICROWAVE_AWG_HANDLER_H_

#include "configs-translator.h"
#include "awg.hpp"
#include "timed-hash-queue.h"
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
    #define MW_MAX_STEP_REPETITION 1048575

    class MicrowaveAwgHandler {

    public:

        struct IQMixerWaveform {
            MicrowaveWaveforms::Waveform waveform;
            double time;
            double t_initial_pause;
            int64_t hash; // store the combined hash
        
            // Constructor computes the hash once
            IQMixerWaveform(const MicrowaveWaveforms::Waveform& wf, double time, double t_initial_pause)
                : waveform(wf), time(time), t_initial_pause(t_initial_pause)
            {
                auto hash_combine = [](int64_t seed, int64_t value) {
                    return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
                };
        
                // Compute hash from waveform variant
                int64_t h = boost::apply_visitor(
                    [](auto&& w) { return w.hash(); }, waveform);
        
                // Combine with other fields
                if (boost::get<MicrowaveWaveforms::Pause>(&wf) == nullptr) { // Not a pause
                    h = hash_combine(h, std::hash<double>{}(time));
                    h = hash_combine(h, std::hash<double>{}(t_initial_pause));
                }
        
                hash = h;
            }
        };

        ConfigsTranslator& config_translator = ConfigsTranslator::instance();
        AWG awg;
        std::unordered_map<int64_t, int> hash_segment_index_map;
        TimedHashQueue awg_segments_queue;

        double dphi, vI_dc, vQ_dc;
        int max_step_size, min_segment_size, max_segment_count, default_pause_segment_size;
        double digital_offset_time;

        std::tuple<std::vector<IQMixerWaveform>, std::vector<int>> breakdown_waveforms(const std::vector<MicrowaveWaveforms::Waveform>& waveforms);

        void upload_iqmixer_waveform(IQMixerWaveform);
        
    public:
        MicrowaveAwgHandler(const std::string& handler_config, const std::string& awg_config);
        ~MicrowaveAwgHandler();

        void open_connection();
        void close_connection();
        void reload(bool flag_translate = true);

        void upload_waveforms(const std::vector<MicrowaveWaveforms::Waveform>& waveforms);

        bool is_connected() const;
    };

}

#endif