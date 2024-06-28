#include "Sequence.h"

void Stream::Sequence::configure() {
    num_total_segments = awg->get_num_segments();
    waveforms_per_segment = awg->get_waveforms_per_segment();
    samples_per_segment = awg->get_samples_per_segment();

    idle_segment_idx = 0;
    null_segment_idx = num_total_segments - 1;
    short_circuit_seg_idx = num_total_segments - 2;

    short_circuit_step = short_circuit_seg_idx * 2 - 1;
    short_circuit_null_step = short_circuit_seg_idx * 2;

    // --- Filling Transfer Buffers with zeros ---
    awg->fill_transfer_buffer(lookup_buffer, samples_per_segment, 0);
    awg->fill_transfer_buffer(upload_buffer, samples_per_segment, 0);
    awg->fill_transfer_buffer(double_sized_buffer, samples_per_segment * 2, 0);
}

int Stream::Sequence::setup(bool setup_idle_segment, int idle_step_idx,
                            bool _2d, int Nt_x, int Nt_y) {
    this->idle_step_idx = idle_step_idx;
    short_circuit_step = idle_step_idx + short_circuit_seg_idx * 2 - 1;
    short_circuit_null_step = idle_step_idx + short_circuit_seg_idx * 2;
    this->setup_idle_segment = setup_idle_segment;
    this->_2d = _2d;
    this->Nt_x = Nt_x;
    this->Nt_y = Nt_y;

    reset(true);
    return 0;
}

int Stream::Sequence::init_segments() {

    int status = 0;

    // Fill the idle_waveforms buffer to be used for the idle, null segments and
    // also padding the last segment
    idle_waveforms.reserve(waveforms_per_segment * awg->get_waveform_length());
    if (awg->get_idle_segment_wfm()) { // depending on user config, either
                                       // fill the buffer with 0s or STATIC
                                       // wfms
        get_static_wfm(idle_waveforms.data(), waveforms_per_segment,
                       Nt_x * Nt_y);
    } else {
        std::fill(idle_waveforms.begin(), idle_waveforms.end(), 0);
    }

    // idle segment init
    if (setup_idle_segment) { // Is LLRS responsible for init-ing the IDLE
                              // segment?
        int num_idle_samples = awg->get_idle_segment_length();
        typename AWG::TransferBuffer idleTB =
            awg->allocate_transfer_buffer(num_idle_samples, false);

        load_idle_wfm(*idleTB, num_idle_samples);
        status |= awg->init_and_load_range(
            *idleTB, num_idle_samples, idle_segment_idx, idle_segment_idx + 1);
    }

    // control segments init
    {
        typename AWG::TransferBuffer tb =
            awg->allocate_transfer_buffer(samples_per_segment, false);
        for (int i = idle_segment_idx + 1; i <= short_circuit_seg_idx; i++) {

            status |= awg->fill_transfer_buffer(tb, samples_per_segment,
                                                i % 2 ? 0x7fff : 0);
            status |=
                awg->init_and_load_range(*tb, samples_per_segment, i, i + 1);
        }
    }

    // double-sized segment init
    {
        int num_double_samples = samples_per_segment * 2;

        typename AWG::TransferBuffer doubleTB =
            awg->allocate_transfer_buffer(num_double_samples, false);

        status |= awg->fill_transfer_buffer(doubleTB, num_double_samples, 0);

        status |= awg->init_and_load_range(*doubleTB, num_double_samples,
                                           short_circuit_seg_idx,
                                           short_circuit_seg_idx + 1);
    }

    // null segment init
    {
        int num_null_samples = awg->get_null_segment_length();

        typename AWG::TransferBuffer nullTB =
            awg->allocate_transfer_buffer(num_null_samples, false);

        load_idle_wfm(*nullTB, num_null_samples);
        std::fill(*nullTB, *nullTB + num_null_samples, -0x7fff);
        status |= awg->init_and_load_range(
            *nullTB, num_null_samples, null_segment_idx, null_segment_idx + 1);
    }
    return status;
}

int Stream::Sequence::init_steps() {
    // make idle step point to itself
    awg->seqmem_update(idle_step_idx, idle_segment_idx, 1, idle_step_idx,
                       SPCSEQ_ENDLOOPALWAYS);

    // set up sequence memory to have even and odd be control and null steps
    // respectively, throughout all AWG memory
    for (int seg_idx = idle_segment_idx + 1; seg_idx <= num_total_segments - 2;
         seg_idx++) {

#ifdef DETECT_NULL // to detect nulls, everything points to the short circuit
                   // null step (which points to itself)
        awg->seqmem_update(seg_idx * 2 - 1 + idle_step_idx, seg_idx, 1,
                           short_circuit_null_step, SPCSEQ_ENDLOOPALWAYS);
#else // DETECT_NULL
      // point control to corresponding null
        awg->seqmem_update(seg_idx * 2 - 1 + idle_step_idx, seg_idx, 1,
                           seg_idx * 2 + idle_step_idx, SPCSEQ_ENDLOOPALWAYS);
#endif
        // point null to itself
        awg->seqmem_update(seg_idx * 2 + idle_step_idx, null_segment_idx, 1,
                           seg_idx * 2 + idle_step_idx, SPCSEQ_ENDLOOPALWAYS);
    }

    // set up sequence memory for short circuit segment
    // point short circuit to it's own null
    awg->seqmem_update(short_circuit_step, short_circuit_seg_idx, 1,
                       short_circuit_null_step, SPCSEQ_ENDLOOPALWAYS);

    // point short circuit null to itself
    awg->seqmem_update(short_circuit_null_step, null_segment_idx, 1,
                       short_circuit_null_step, SPCSEQ_ENDLOOPALWAYS);

    return 0;
}

void Stream::Sequence::reset(bool reset_segments) {
    if (reset_segments) {
        init_segments();
    }
    init_steps();

    move_idx = 0;
    load_seg_idx = 0;
    old_control = 0;
    new_control = 0;
    old_null = 0;
    played_first_seg = 0;
}

bool Stream::Sequence::load_single_segment(
    std::vector<Reconfig::Move> &moves_list) {

    move_idx = 0;

    // lookup all waveforms and load it into the transfer buffer
    START_TIMER("V-Latency");
    START_TIMER("V-First-Lookup");
    wf_segment_lookup(*double_sized_buffer, moves_list,
                      waveforms_per_segment * 2);
    END_TIMER("V-First-Lookup");

    START_TIMER("V-First-Upload");
    // upload the segment
    awg->load_data(short_circuit_seg_idx, *double_sized_buffer,
                   samples_per_segment * 2 * sizeof(short));
    END_TIMER("V-First-Upload");
    END_TIMER("V-Latency");
    GET_EXTERNAL_TIME("V-First-Update", 0);
    GET_EXTERNAL_TIME("V-Second-Lookup", 0);
    GET_EXTERNAL_TIME("V-Second-Upload", 0);

    // sets last segment (double-size segment) to be loaded to point to idle
    awg->seqmem_update(short_circuit_step, short_circuit_seg_idx, 1,
                       idle_step_idx, SPCSEQ_ENDLOOPALWAYS);

    // point idle to the double sized segment
    awg->seqmem_update(idle_step_idx, idle_segment_idx, 1, short_circuit_step,
                       SPCSEQ_ENDLOOPALWAYS);
    START_TIMER("V-Load_Stream");

    while (awg->get_current_step() == idle_step_idx) {
    }
    awg->seqmem_update(idle_step_idx, idle_segment_idx, 1, idle_step_idx,
                       SPCSEQ_ENDLOOPALWAYS);
    while (awg->get_current_step() != idle_step_idx) {
    }
    END_TIMER("V-Load_Stream");

    return 0;
}

bool Stream::Sequence::load_multiple_segments(
    std::vector<Reconfig::Move> &moves_list) {

    int num_moves = moves_list.size();
    move_idx = 0;
    played_first_seg = 0;
    load_seg_idx = 1;

    // find number of segments to be loaded
    int extra_moves = num_moves % waveforms_per_segment;
    int num_whole_segments = num_moves / waveforms_per_segment;
    num_segments_to_load = num_whole_segments + (extra_moves != 0);
    int sample_filling_required = extra_moves == 0
                                      ? 0
                                      : (waveforms_per_segment - extra_moves) *
                                            awg->get_waveform_length();

    // sets last segment to be loaded to point to idle
    last_control_step = idle_step_idx + (num_segments_to_load * 2) - 1;
    awg->seqmem_update(last_control_step,
                       num_segments_to_load + idle_segment_idx, 1,
                       idle_step_idx, SPCSEQ_ENDLOOPALWAYS);

    short *lookup_pointer = *lookup_buffer;
    short *upload_pointer = *upload_buffer;

    // lookup first segment
    START_TIMER("V-Latency");
    START_TIMER("V-First-Lookup");
    wf_segment_lookup(lookup_pointer, moves_list, waveforms_per_segment);
    END_TIMER("V-First-Lookup");

    START_TIMER("V-First-Upload");
    // swap buffers
    std::swap(lookup_pointer, upload_pointer);

    // pre - load first segment
    awg->load_data_async_start(idle_segment_idx + 1, upload_pointer,
                               samples_per_segment * sizeof(short));

    // lookup second segment
    START_TIMER("V-Second-Lookup");
    wf_segment_lookup(lookup_pointer, moves_list, waveforms_per_segment);
    END_TIMER("V-Second-Lookup");

    // wait for old transfer to finish
    awg->wait_for_data_load();
    END_TIMER("V-First-Upload");

    START_TIMER("V-Second-Upload");
    // load all segments
    for (load_seg_idx = idle_segment_idx + 2;
         load_seg_idx <= num_segments_to_load + idle_segment_idx;
         load_seg_idx++) {

        // swap buffers
        std::swap(lookup_pointer, upload_pointer);
        // ---upload data---
        awg->load_data_async_start(load_seg_idx, upload_pointer,
                                   samples_per_segment * sizeof(short));

        //---lookup waveforms for next segment
        wf_segment_lookup(lookup_pointer, moves_list, waveforms_per_segment);
        if (load_seg_idx == num_segments_to_load + idle_segment_idx) {
            load_idle_wfm(lookup_pointer +
                              extra_moves * awg->get_waveform_length(),
                          sample_filling_required);
        }

        // wait for old upload to finish
        awg->wait_for_data_load();

        if (load_seg_idx == idle_segment_idx + 2) {
            END_TIMER("V-Second-Upload");
            START_TIMER("V-First-Update");
        }

        // ---update sequence memory---
        old_control = idle_step_idx + load_seg_idx * 2 - 3;
        old_null = idle_step_idx + load_seg_idx * 2 - 2;
        new_control = idle_step_idx + load_seg_idx * 2 - 1;

        // point old control to new control
        awg->seqmem_update(old_control, load_seg_idx - 1, 1, new_control,
                           SPCSEQ_ENDLOOPALWAYS);

#ifndef DETECT_NULL
        // point old null to new control
        awg->seqmem_update(old_null, null_segment_idx, 1, new_control,
                           SPCSEQ_ENDLOOPALWAYS);
#endif
        // if it's the second segment that just loaded, take playback off
        // idle
        if (load_seg_idx == (idle_segment_idx + 2)) {
            awg->seqmem_update(idle_step_idx, idle_segment_idx, 1, old_control,
                               SPCSEQ_ENDLOOPALWAYS);
            END_TIMER("V-First-Update");
            END_TIMER("V-Latency");
            START_TIMER("V-Load_Stream");
        }
    }

#ifdef DETECT_NULL
    while (awg->get_current_step() == idle_step_idx) {
    }
    awg->seqmem_update(idle_step_idx, idle_segment_idx, 1, idle_step_idx,
                       SPCSEQ_ENDLOOPALWAYS);
    while (awg->get_current_step() != idle_step_idx) {
        if (awg->get_current_step() == short_circuit_null_step) {
            std::cerr << "Detected null" << std::endl;
            awg->seqmem_update(idle_step_idx, idle_segment_idx, 1,
                               idle_step_idx, SPCSEQ_ENDLOOPALWAYS);
            awg->seqmem_update(short_circuit_null_step, idle_segment_idx, 1,
                               idle_step_idx, SPCSEQ_ENDLOOPALWAYS);
            while (awg->get_current_step() != idle_step_idx) {
            }
            return 1;
        }
    }
#else // DETECT_NULL

    while (awg->get_current_step() == idle_step_idx) {
    }
    awg->seqmem_update(idle_step_idx, idle_segment_idx, 1, idle_step_idx,
                       SPCSEQ_ENDLOOPALWAYS);
    while (awg->get_current_step() != idle_step_idx) {
    }

#endif
    END_TIMER("V-Load_Stream");

    return 0;
}

void Stream::Sequence::wf_segment_lookup(
    short *p_buffer_lookup, std::vector<Reconfig::Move> &moves_list,
    int waveforms_per_segment) {

    for (int wf_idx = 0;
         wf_idx < waveforms_per_segment && move_idx < moves_list.size();
         wf_idx++, move_idx++) {
        Reconfig::Move move = moves_list[move_idx];
        short *move_wf_ptr = wf_table.get_waveform_ptr(
            std::get<0>(move), std::get<4>(move), std::get<1>(move),
            std::get<2>(move), std::get<3>(move));
        memcpy(p_buffer_lookup + wf_idx * awg->get_waveform_length(),
               move_wf_ptr, awg->get_waveform_length() * sizeof(short));
    }
}

void Stream::Sequence::get_static_wfm(int16 *pnData, size_t num_wfms,
                                      int Nt_x) {

    short *move_wf_ptr = wf_table.get_primary_wf_ptr(STATIC, 0, Nt_x, Nt_x);
    auto wfm_length = awg->get_waveform_length();
    for (int i = 0; i < num_wfms; ++i) {
        memcpy(pnData + i * wfm_length, move_wf_ptr,
               wfm_length * sizeof(short));
    }
}

void Stream::Sequence::load_idle_wfm(short *p_buffer, int num_samples) {
    if (idle_waveforms.size() >= num_samples) {
        memcpy(p_buffer, idle_waveforms.data(), num_samples * sizeof(short));
    } else {
        if (awg->get_idle_segment_wfm()) { // depending on user config, either
                                           // fill the buffer with 0s or STATIC
                                           // wfms
            get_static_wfm(p_buffer, num_samples / awg->get_waveform_length(),
                           Nt_x * Nt_y);
        } else {
            std::fill(p_buffer, p_buffer + num_samples, 0);
        }
    }
}
