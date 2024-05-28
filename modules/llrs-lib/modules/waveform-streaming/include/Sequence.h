#ifndef AWGSEQUENCE_H_
#define AWGSEQUENCE_H_

#define LOAD_SINGLE_SEGMENT

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
    Sequence(Util::Collector *p_collector, Synthesis::WaveformTable &wf_table,
             double waveform_duration)
        : awg{std::make_shared<AWG>()}, p_collector{p_collector},
          wf_table{wf_table}, lookup_buffer{awg->allocate_transfer_buffer(
                                  awg->get_samples_per_segment(), false)},
          upload_buffer{awg->allocate_transfer_buffer(
              awg->get_samples_per_segment(), false)},
          double_sized_buffer{awg->allocate_transfer_buffer(
              awg->get_samples_per_segment() * 2, false)} {

        awg->configure_segment_length(waveform_duration);
        lookup_buffer =
            awg->allocate_transfer_buffer(awg->get_samples_per_segment(), true);
        upload_buffer =
            awg->allocate_transfer_buffer(awg->get_samples_per_segment(), true);
        double_sized_buffer = awg->allocate_transfer_buffer(
            awg->get_samples_per_segment() * 2, true);
        configure();
    }

    Sequence(std::shared_ptr<AWG_T> &awg, Util::Collector *p_collector,
             Synthesis::WaveformTable &wf_table, double waveform_duration)
        : awg{awg}, p_collector{p_collector}, wf_table{wf_table},
          lookup_buffer{awg->allocate_transfer_buffer(
              awg->get_samples_per_segment(), false)},
          upload_buffer{awg->allocate_transfer_buffer(
              awg->get_samples_per_segment(), false)},
          double_sized_buffer{awg->allocate_transfer_buffer(
              awg->get_samples_per_segment() * 2, false)} {

        awg->configure_segment_length(waveform_duration);
        lookup_buffer =
            awg->allocate_transfer_buffer(awg->get_samples_per_segment(), true);
        upload_buffer =
            awg->allocate_transfer_buffer(awg->get_samples_per_segment(), true);
        double_sized_buffer = awg->allocate_transfer_buffer(
            awg->get_samples_per_segment() * 2, true);
        configure();
    }

    int setup(bool setup_idle_segment, int idle_step_idx, bool _2d, int Nt_x,
              int Nt_y);
    bool load_and_stream(std::vector<Reconfig::Move> &moves_list, int trial_num,
                         int rep_num, int cycle_num) {
        return moves_list.size() <= waveforms_per_segment * 2
                   ? load_single_segment(moves_list, trial_num, rep_num,
                                         cycle_num)
                   : load_multiple_segments(moves_list, trial_num, rep_num,
                                            cycle_num);
    }
    bool load_single_segment(std::vector<Reconfig::Move> &moves_list,
                             int trial_num, int rep_num, int cycle_num);
    bool load_multiple_segments(std::vector<Reconfig::Move> &moves_list,
                                int trial_num, int rep_num, int cycle_num);
    void emccd_trigger() { awg->generate_async_output_pulse(EMCCD); }
    void reset(bool reset_segments);

    void get_static_wfm(int16 *pnData, size_t num_wfms, int Nt_x);
    double get_waveform_duration() const {
        return awg->get_waveform_duration();
    }
    double get_sample_rate() const { return awg->get_sample_rate(); }
    int get_waveform_length() const { return awg->get_waveform_length(); }
    int get_waveform_mask() const { return awg->get_wavefrom_mask(); }
    int get_vpp() const { return awg->get_vpp(); }
    int get_wfm_per_segment() const { return awg->get_waveforms_per_segment(); }
    int get_acq_timeout() const { return awg->get_acq_timeout(); }
    int get_current_step() const { return awg->get_current_step(); }
    void start_stream() { awg->start_stream(); }

  private:
    std::shared_ptr<AWG_T> awg;
    Util::Collector *p_collector;
    Synthesis::WaveformTable &wf_table;

    bool setup_idle_segment = true;
    int waveforms_per_segment;
    int samples_per_segment;

    int num_total_segments;
    int num_segments_to_load;
    int idle_segment_idx;
    int short_circuit_seg_idx;
    int null_segment_idx;

    int idle_step_idx = 0;
    int last_control_step;
    int short_circuit_null_step;
    int short_circuit_step;

    typename AWG_T::TransferBuffer lookup_buffer;
    typename AWG_T::TransferBuffer upload_buffer;
    typename AWG_T::TransferBuffer double_sized_buffer;

    size_t move_idx;
    size_t load_seg_idx;
    size_t old_control;
    size_t new_control;
    size_t old_null;
    bool played_first_seg;

    bool _2d;
    int Nt_x;
    int Nt_y;

    void configure();
    int init_segments();
    int init_steps();
    void load_idle_wfm(short *p_buffer, int num_samples);
    void wf_segment_lookup(short *p_buffer_lookup,
                           std::vector<Reconfig::Move> &moves_list,
                           int waveforms_per_segment);
};

} // namespace Stream

#endif
