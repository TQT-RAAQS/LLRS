#ifndef AWGSEQUENCE_H_
#define AWGSEQUENCE_H_

#include "Collector.h"
#include "Setup.h"
#include "Solver.h"
#include "WaveformTable.h"
#include "awg.hpp"
#include "llrs-lib/Settings.h"
#include <memory>
#include <vector>

namespace Stream {

template <typename AWG_T> class Sequence {
  public:
    /**
     * @brief Construct a new Sequence object
     * @param p_collector The pointer to the collector singleton object
     * @param wf_table  The reference to the waveform table object
     */
    Sequence(Util::Collector *p_collector, Synthesis::WaveformTable &wf_table)
        : p_collector{p_collector}, wf_table{wf_table} {}

    Sequence(std::shared_ptr<AWG_T> &awg, Util::Collector *p_collector,
             Synthesis::WaveformTable &wf_table)
        : awg{awg}, p_collector{p_collector}, wf_table{wf_table} {}

    void setup(int idle_segment_idx, int idle_step_idx, bool _2d, int Nt_x,
               int Nt_y);
    void pre_load();
    bool load_and_stream(std::vector<Reconfig::Move> &moves_list, int trial_num,
                         int rep_num, int cycle_num);
    void emccd_trigger() { awg->generate_async_output_pulse(ASYNC_TRIG_AMP); }
    void reset();

    void get_1d_static_wfm(int16 *pnData, int num_wfms, int Nt_x);
    double get_waveform_duration() const { awg->get_waveform_duration(); }
    double get_sample_rate() const { awg->get_sample_rate(); }
    int get_waveform_length() const { awg->get_waveform_length(); }
    int get_waveform_mask() const { awg->get_wavefrom_mask(); }
    int get_vpp() const { awg->get_vpp(); }
    int get_wfm_per_segment() const { awg->get_waveforms_per_segment(); }

  private:
    std::shared_ptr<AWG_T> awg;
    Util::Collector *p_collector;
    Synthesis::WaveformTable &wf_table;

    int idle_segment_idx = 0;
    int idle_step_idx = 0;
    int num_total_segments;
    int waveforms_per_segment;
    int samples_per_segment;
    int short_circuit_seg_idx;
    int null_segment_idx;
    int num_segments_to_load;
    size_t last_control_step;
    int short_circuit_null_step;
    int short_circuit_step;
    int current_step;
    short *p_contbuf_one;
    short *p_buffer_lookup;
    short *p_buffer_upload;
    int16 *p_buffer_double;
    int buffer_lookup_size;
    int buffer_upload_size;
    int buffer_double_size;
    size_t move_idx;
    size_t load_seg_idx;
    size_t old_control;
    size_t new_control;
    size_t old_null;
    bool played_first_seg;

    void wf_segment_lookup(short *p_buffer_lookup,
                           std::vector<Reconfig::Move> &moves_list,
                           int waveforms_per_segment);
};

} // namespace Stream

#endif
