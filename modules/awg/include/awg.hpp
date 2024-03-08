#ifndef STREAM_AWG_HPP
#define STREAM_AWG_HPP

#include "common.hpp"
#include "spcm_includes.h"

#define MAX_SEGMENT_DIVIDE 4096

class AWG { 
public: 
        AWG();
        ~AWG();

        uint32 enable_channels(const std::vector<int>& channels);
        uint32 enable_outputs(const std::vector<int>& channels, const std::vector<int>& amp);
        uint32 set_sample_rate(int sample_rate);
        uint32 set_external_clock_mode(int external_clock_freq);
        uint32 set_internal_clock_mode(int external_clock_freq);
        uint32 set_trigger_settings();
        uint32 set_dout_trigger_mode(int32 line, int32 channel);
        uint32 set_dout_async(int32 line);
        uint32 generate_async_output_pulse(int32 voltage);

        uint32 start_stream();
        uint32 reset_card();
        uint32 stop_card();
        void close_card();

        drv_handle get_card() const { return p_card; };
        int get_num_channels() const { return num_channels; };
        int get_sample_rate() const { return config.sample_rate; };
        int get_waveform_duration() const { return config.waveform_duration; };
        int get_num_segments() const { return config.awg_num_segments; };
        int get_waveforms_per_segment() const { return config.waveforms_per_segment; };
        int get_samples_per_segment() const { return config.samples_per_segment; };

        uint32 seqmem_update(int64 lStep, int64 llSegment, int64 llLoop, int64 llNext, uint64 llCondition);
        uint32 load_data(uint32 seg_num, short* p_data, uint64 size);
        uint32 load_data_async_start(uint32 seg_num, short* p_data, uint64 size);
        uint32 init_segment(uint32 seg_num, int num_samples);
        uint32 init_and_load_all(short * p_segment, int num_samples);
        uint32 init_and_load_range(short * p_segment, int num_samples, int start, int end);
        uint32 wait_for_data_load();
        uint32 get_current_step();
        uint32 get_last_seg() const { return this->max_seg - 1; };
        uint32 get_last_step() const { return this->max_step - 1; };
        int allocate_transfer_buffer( int num_samples, int16 *&pnData );
        uint32 fill_transfer_buffer( int16 *pnData, int num_samples, int16 value );
        void print_awg_error();
        uint32 read_config(std::string filename);

private:
    struct awg_config_t {
        char*               driver_path;
        int                 sample_rate;
        int                 external_clock_freq;
        int                 freq_resolution;
        std::vector<int>    channels;
        std::vector<int>    amp;
        int                 amp_lim;
        int                 awg_num_segments;
        double              awg_sample_rate;
        int                 wfm_mask;
        double              waveform_duration;
        int                 waveforms_per_segment;
        int                 null_segment_length;
        int                 idle_segment_length;
        int                 waveform_length;
        int                 samples_per_segment;

        ~awg_config_t() {
            delete[] driver_path;
        }
    };
        awg_config_t config; 
        drv_handle p_card;
        int num_channels;
        uint32 max_step     = 0;
        uint32 max_seg      = 0;
        int bps             = 0;
        int lSetChannels    = 0;
        int dwFactor        = 0;

        uint32 configure();

};
#endif

