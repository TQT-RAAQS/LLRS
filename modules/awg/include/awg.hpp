#ifndef AWG_HPP_
#define AWG_HPP_

#include <boost/variant.hpp>
#include <tuple>
#include "common.hpp"
#include "spcm_includes.h"
#include <unordered_map>
#include <algorithm>


#define AWG_MEMORY_SIZE 4294967296

enum TriggerType { 
  X0=1, 
  X1=2,
  X2=4
};

class AWG {
  public:
    AWG(std::string config_name = "trapping.yml");
    ~AWG();

    int open_connection();
    int start_stream();
    int reset_card();
    int stop_card();
    void close_card();

    bool is_connection_open() const { return flag_is_connected; }

    void force_hardware_trigger();
    void configure_segment_length(double waveform_duration);
    int seqmem_update(int64 lStep, int64 llSegment, int64 llLoop, int64 llNext,
                      uint64 llCondition);

    void interleave_data(short* target, const std::vector<std::vector<short>> &waveforms, const std::vector<std::vector<int8>>& digital_trigger = {});
    int load_data(int seg_num, short *p_data, uint64 size, bool wait_until_finished = true);
    int init_segment(int seg_num, int num_samples);
    int init_and_load_all(short *p_segment, int num_samples);
    int init_and_load_range(short *p_segment, int num_samples, int start,
                            int end);
    int wait_for_data_load();
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
    int get_waveform_length() const { return config.waveform_length; };
    int get_null_segment_length() const { return config.null_segment_length; };
    int get_idle_segment_length() const { return config.idle_segment_length; };
    int get_wavefrom_mask() const { return config.wfm_mask; };
    int get_current_step();
    int get_last_seg() const { return config.awg_num_segments - 1; };
    int get_last_step() const { return max_step - 1; };
    bool get_idle_segment_wfm() const { return config.idle_segment_wfm; }
    void print_awg_error();
    std::tuple<int, std::string> get_awg_error();

    class TransferBuffer {
        void *buffer;
        size_t size;
        bool contBuf; // Is it a physically continuous buffer? See Spectrum's
                      // "Continuous memory for increased data transfer rate"
                      // feature

        TransferBuffer(AWG &awg, size_t size, bool contBuf = false);

      public:
        short *operator*() { return (short *)buffer; }
        TransferBuffer(const TransferBuffer &other) = delete;
        TransferBuffer &operator=(const TransferBuffer &other) = delete;
        TransferBuffer(TransferBuffer &&other);
        TransferBuffer &operator=(TransferBuffer &&other);
        ~TransferBuffer();
        friend class AWG;
    };
    TransferBuffer allocate_transfer_buffer(int num_samples,
                                            bool contBuf = false);
    int fill_transfer_buffer(TransferBuffer &tb, int num_samples, int16 value);

  private:
    bool flag_is_connected = false;

    struct input_trigger_config_t {
        std::vector<size_t> ports;
        uint8_t logic;
        uint8_t edge;
        uint8_t rearm;
        std::vector<int32> level_0_mv;
        std::vector<int32> level_1_mv;
        int32 timeout_ms;
    };

    struct sync_output_trigger_config_t {
        int8_t port;
        int8_t channel;
        int8_t bit;
    };

    int set_sample_rate(int sample_rate);
    int set_external_clock_mode(int external_clock_freq);
    int set_internal_clock_mode();
    int set_input_trigger_settings(const input_trigger_config_t& config);
    int set_dout_trigger_mode(int32 line, int32 channel);
    int setup_async_output_triggers(std::vector<int8_t> channels);
    int setup_sync_output_triggers(std::vector<sync_output_trigger_config_t> configs);
    int read_config(std::string filename);
    int enable_channels(const std::vector<int> &channels);
    int enable_outputs(const std::vector<int> &channels,
                       const std::vector<int> &amp);

    std::string config_name;
    struct awg_config_t {
        std::string driver_path;
        bool external_clock_flag;
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
        bool idle_segment_wfm;
        int null_seg_num_waveforms;
        int idle_seg_num_waveforms;
        input_trigger_config_t input_trigger_config;
        int32 first_step_index;
        std::vector<int8_t> async_out_trig_channels;
        std::vector<sync_output_trigger_config_t> sync_out_trig_configs;
        std::unordered_map<int, int> channel_bit_shifts;
        std::unordered_map<int, std::vector<int>> channel_digout_indices;

        ~awg_config_t() { }
    } config;
    drv_handle p_card;
    int num_channels;
    int max_step;
    int bps;
    int lSetChannels;
    int dwFactor = 1;
    void *continuousBuffer = nullptr;
    uint64 continuousBufferSize = 0;
    friend class TransferBuffer;
};

#endif
