#ifndef AWG_HPP_
#define AWG_HPP_

#include "common.hpp"
#include "spcm_includes.h"

#define AWG_MEMORY_SIZE 4294967296

enum TriggerType { EMCCD };

class AWG {
  public:
    AWG();
    ~AWG();

    int start_stream();
    int reset_card();
    int stop_card();
    void close_card();

    int seqmem_update(int64 lStep, int64 llSegment, int64 llLoop, int64 llNext,
                      uint64 llCondition);
    int load_data(int seg_num, short *p_data, uint64 size);
    int load_data_async_start(int seg_num, short *p_data, uint64 size);
    int init_segment(int seg_num, int num_samples);
    int init_and_load_all(short *p_segment, int num_samples);
    int init_and_load_range(short *p_segment, int num_samples, int start,
                            int end);
    void setup_segment_memory(int16 *pnData);
    int wait_for_data_load();
    int allocate_transfer_buffer(int num_samples, int16 *&pnData);
    int fill_transfer_buffer(int16 *pnData, int num_samples, int16 value);
    void generate_async_output_pulse(TriggerType type);

    /// Getters
    int get_num_channels() const { return num_channels; };
    double get_sample_rate() const { return config.sample_rate; };
    double get_waveform_duration() const { return config.waveform_duration; };
    int get_num_segments() const { return config.awg_num_segments; };
    int get_waveforms_per_segment() const {
        return config.waveforms_per_segment;
    };
    int get_samples_per_segment() const { return config.samples_per_segment; };
    int get_trigger_size() const { return config.trigger_size; };
    int get_vpp() const { return config.vpp; };
    int get_acq_timeout() const { return config.acq_timeout; };
    int get_async_trig_amp() const { return config.async_trig_amp; };
    int get_waveform_length() const { return config.waveform_length; };
    int get_null_segment_length() const { return config.null_segment_length; };
    int get_idle_segment_length() const { return config.idle_segment_length; };
    int get_wavefrom_mask() const { return config.wfm_mask; };
    int get_current_step();
    int get_last_seg() const { return config.awg_num_segments - 1; };
    int get_last_step() const { return max_step - 1; };
    void print_awg_error();

  private:
    int set_sample_rate(int sample_rate);
    int set_external_clock_mode(int external_clock_freq);
    int set_internal_clock_mode();
    int set_trigger_settings();
    int set_dout_trigger_mode(int32 line, int32 channel);
    int set_dout_async(int32 line);
    int read_config(std::string filename);
    int enable_channels(const std::vector<int> &channels);
    int enable_outputs(const std::vector<int> &channels,
                       const std::vector<int> &amp);

    struct awg_config_t {
        char *driver_path;
        int external_clock_freq;
        std::vector<int> channels;
        std::vector<int> amp;
        int awg_num_segments;
        double sample_rate;
        double waveform_duration;
        int wfm_mask;
        int waveforms_per_segment;
        int null_segment_length;
        int idle_segment_length;
        int waveform_length;
        int samples_per_segment;
        int trigger_size;
        int vpp;
        int acq_timeout;
        int async_trig_amp;
        ~awg_config_t() { delete[] driver_path; }
    } config;
    drv_handle p_card;
    int num_channels;
    int max_step;
    int bps;
    int lSetChannels;
    int dwFactor;
    int configure();
};
#endif
