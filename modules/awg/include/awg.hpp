#ifndef AWG_HPP_
#define AWG_HPP_

#include "common.hpp"
#include "spcm_includes.h"

#define MAX_SEGMENT_DIVIDE 4096

enum TriggerType { EMCCD };

class AWG {
  public:
    AWG();
    ~AWG();

    uint32 start_stream();
    uint32 reset_card();
    uint32 stop_card();
    void close_card();

    uint32 set_sample_rate(int sample_rate);
    uint32 set_external_clock_mode(int external_clock_freq);
    uint32 set_internal_clock_mode();
    uint32 set_trigger_settings();
    uint32 set_dout_trigger_mode(int32 line, int32 channel);
    uint32 set_dout_async(int32 line);

    drv_handle get_card() const { return p_card; };
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

    uint32 get_current_step();
    uint32 get_last_seg() const { return this->max_seg - 1; };
    uint32 get_last_step() const { return this->max_step - 1; };

    uint32 seqmem_update(int64 lStep, int64 llSegment, int64 llLoop,
                         int64 llNext, uint64 llCondition);
    uint32 load_data(uint32 seg_num, short *p_data, uint64 size);
    uint32 load_data_async_start(uint32 seg_num, short *p_data, uint64 size);
    uint32 init_segment(uint32 seg_num, int num_samples);
    uint32 init_and_load_all(short *p_segment, int num_samples);
    uint32 init_and_load_range(short *p_segment, int num_samples, int start,
                               int end);
    void setup_segment_memory(int16 *pnData);
    uint32 wait_for_data_load();
    int allocate_transfer_buffer(int num_samples, int16 *&pnData);
    uint32 fill_transfer_buffer(int16 *pnData, int num_samples, int16 value);
    void print_awg_error();
    uint32 read_config(std::string filename);
    uint32 enable_channels(const std::vector<int> &channels);
    uint32 enable_outputs(const std::vector<int> &channels,
                          const std::vector<int> &amp);
    uint32 generate_async_output_pulse(TriggerType type);

  private:
    struct awg_config_t {
        char *driver_path;
        int external_clock_freq;
        std::vector<int> channels;
        std::vector<int> amp;
        int awg_num_segments;
        double sample_rate;
        int wfm_mask;
        double waveform_duration;
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
    };
    awg_config_t config;
    drv_handle p_card;
    int num_channels;
    uint32 max_step = 0;
    uint32 max_seg = 0;
    int bps = 0;
    int lSetChannels = 0;
    int dwFactor = 0;

    uint32 configure();
};
#endif
