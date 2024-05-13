#include "Sequence.h"

template <typename AWG_T> void Stream::Sequence<AWG_T>::configure() {
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
    awg->fill_transfer_buffer(double_size_buffer, samples_per_segment * 2, 0);
}

template <typename AWG_T>
int Stream::Sequence<AWG_T>::setup(bool setup_idle_segment, int idle_step_idx,
                                   bool _2d, int Nt_x, int Nt_y) {
    this->idle_step_idx = idle_step_idx;
    this->setup_idle_segment = setup_idle_segment;

    reset();
}

template <typename AWG_T> int init_segments() {

    int status = 0;

    // idle segment init
    if (setup_idle_segment) { // Is LLRS responsible for init-ing the IDLE
                              // segment?
        int num_idle_samples = awg->get_idle_segment_length();
        AWG_T::TransferBuffer idleTB =
            awg->allocate_transfer_buffer(num_idle_samples, false);

        if (awg->get_idle_segment_wfm()) { // depending on user config, either
                                           // fill the buffer with 0s or STATIC
                                           // wfms
            get_static_wfm(*idleTB,
                           num_idle_samples / awg->get_waveform_length(),
                           Nt_x * Nt_y);
        } else {
            status |= awg->fill_transfer_buffer(idleTB, num_idle_samples, 0);
        }

        status |= awg->init_and_load_range(
            *idleTB, num_idle_samples, idle_segment_idx, idle_segment_idx + 1);
    }

    // control segments init
    {
        AWG_T::TransferBuffer tb =
            awg->allocate_transfer_buffer(samples_per_segment, false);

        status |= awg->fill_transfer_buffer(tb, samples_per_segment, 0);

        status |= awg->init_and_load_range(
            *tb, samples_per_segment, idle_segment_idx, short_circuit_seg_idx);
    }

    // double-sized segment init
    {
        int num_double_samples = samples_per_segment * 2;

        AWG_T::TransferBuffer doubleTB =
            awg->allocate_transfer_buffer(num_double_samples, false);

        status |= awg->fill_transfer_buffer(doubleTB, num_double_samples, 0);

        status |= awg->init_and_load_range(doubleTB, num_double_samples,
                                           short_circuit_seg_idx,
                                           short_circuit_seg_idx + 1);
    }

    // null segment init
    {
        int NULL_MASK = 1 << 15; // 15th bit set to 1 for null segment counter
        int num_null_samples = awg->get_null_segment_length();

        AWG_T::TransferBuffer nullTB =
            awg->allocate_transfer_buffer(num_null_samples, false);

        awg->fill_transfer_buffer(nullTB, num_null_samples, 0);

        // make half the samples have the hardware trigger
        for (int null_index = num_null_samples / 2;
             null_index < num_null_samples; null_index++) {
            if (_2d) {
                // in 2 channels, we arm every other sample otherwise triggers
                // aren't consistent
                if (null_index % 2 == 0) {
                    (*nullTB)[null_index] =
                        NULL_MASK | null_segment_data[null_index];
                }
            } else {
                null_segment_data[null_index] =
                    NULL_MASK | (*nullTB)[null_index];
            }
        }

        awg->init_and_load_range(*nullTB, num_null_samples, null_segment_idx,
                                 null_segment_idx + 1);
    }
}

template <typename AWG_T> int init_steps() {
    // make idle step point to itself
    awg->seqmem_update(idle_step_idx, idle_segment_idx, 1, idle_step_idx,
                       SPCSEQ_ENDLOOPALWAYS);

    // set up sequence memory to have even and odd be control and null steps
    // respectively, throughout all AWG memory
    for (int seg_idx = idle_segment_idx + 1; seg_idx <= num_total_segments - 2;
         seg_idx++) {

        // point control to corresponding null
        awg->seqmem_update(seg_idx * 2 - 1 + idle_step_idx, seg_idx, 1,
                           seg_idx * 2 + idle_step_idx, SPCSEQ_ENDLOOPALWAYS);

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
}

template <typename AWG_T> void Stream::Sequence<AWG_T>::reset() {
    init_segments();
    init_steps();

    move_idx = 0;
    load_seg_idx = 0;
    old_control = 0;
    new_control = 0;
    old_null = 0;
    played_first_seg = 0;
}

template <typename AWG_T>
bool Stream::Sequence<AWG_T>::load_and_stream(
    std::vector<Reconfig::Move> &moves_list, int trial_num, int rep_num,
    int cycle_num) {

    bool finished = false;
    move_idx = 0;
    played_first_seg = 0;
    load_seg_idx = 1;

    if (num_moves <= waveforms_per_segment * 2) {

        // sets last segment (double-size segment) to be loaded to point to idle
        awg->seqmem_update(short_circuit_step, short_circuit_seg_idx, 1,
                           idle_step_idx, SPCSEQ_ENDLOOPALWAYS);

#ifdef LOGGING_VERBOSE
        INFO << "Started single segment loading using double-sized segment"
             << std::endl;
#endif

        // lookup all waveforms and load it into the transfer buffer
#ifdef LOGGING_RUNTIME
        p_collector->start_timer("V-Load_Stream", trial_num, rep_num,
                                 cycle_num);
        p_collector->start_timer("V-Latency", trial_num, rep_num, cycle_num);
        p_collector->start_timer("V-First-Lookup", trial_num, rep_num,
                                 cycle_num);
#endif

        wf_segment_lookup(*double_size_buffer, moves_list,
                          waveforms_per_segment * 2);

#ifdef LOGGING_RUNTIME
        p_collector->end_timer("V-First-Lookup", trial_num, rep_num, cycle_num);
        p_collector->start_timer("V-First-Upload", trial_num, rep_num,
                                 cycle_num);
#endif

        // upload the segment
        awg->load_data_async_start(short_circuit_seg_idx, *double_sized_buffer,
                                   samples_per_segment * 2 * sizeof(short));

        // point idle to the double sized segment
        awg->seqmem_update(idle_step_idx, idle_segment_idx, 1,
                           short_circuit_step, SPCSEQ_ENDLOOPALWAYS);

        awg->wait_for_data_load();

#ifdef LOGGING_RUNTIME
        p_collector->end_timer("V-First-Upload", trial_num, rep_num, cycle_num);
        p_collector->start_timer("V-First-Update", trial_num, rep_num,
                                 cycle_num);
        p_collector->end_timer("V-First-Update", trial_num, rep_num, cycle_num);
        p_collector->end_timer("V-Latency", trial_num, rep_num, cycle_num);
        p_collector->get_external_time("V-Second-Lookup", trial_num, rep_num,
                                       cycle_num, 0);
        p_collector->get_external_time("V-Second-Upload", trial_num, rep_num,
                                       cycle_num, 0);
#endif

    } else { /// MULTI-SEGMENT LOADING

        // find number of segments to be loaded
        int num_moves = moves_list.size();
        int extra_moves = num_moves % waveforms_per_segment;
        int num_whole_segments = num_moves / waveforms_per_segment;
        num_segments_to_load = num_whole_segments + (extra_moves != 0);

        // sets last segment to be loaded to point to idle
        last_control_step = idle_step_idx + num_segments_to_load * 2 - 1;
        awg->seqmem_update(last_control_step, num_segments_to_load, 1,
                           idle_segment_idx, SPCSEQ_ENDLOOPALWAYS);

        short *lookup_pointer = *lookup_buffer;
        short *upload_pointer = *upload_buffer;

        // lookup first segment
#ifdef LOGGING_RUNTIME
        p_collector->start_timer("V-Load_Stream", trial_num, rep_num,
                                 cycle_num);
        p_collector->start_timer("V-Latency", trial_num, rep_num, cycle_num);
        p_collector->start_timer("V-First-Lookup", trial_num, rep_num,
                                 cycle_num);
#endif

        wf_segment_lookup(lookup_pointer, moves_list, waveforms_per_segment);

#ifdef LOGGING_RUNTIME
        p_collector->end_timer("V-First-Lookup", trial_num, rep_num, cycle_num);
        p_collector->start_timer("V-First-Upload", trial_num, rep_num,
                                 cycle_num);
#endif

        // swap buffers
        std::swap(lookup_pointer, upload_pointer);

        // pre - load first segment
        awg->load_data_async_start(idle_segment_idx + 1, p_buffer_upload,
                                   samples_per_segment * sizeof(short));

        // lookup second segment
#ifdef LOGGING_RUNTIME
        p_collector->start_timer("V-Second-Lookup", trial_num, rep_num,
                                 cycle_num);
#endif

        wf_segment_lookup(lookup_pointer, moves_list, waveforms_per_segment);

#ifdef LOGGING_RUNTIME
        p_collector->end_timer("V-Second-Lookup", trial_num, rep_num,
                               cycle_num);
#endif

        // wait for old transfer to finish
        awg->wait_for_data_load();

#ifdef LOGGING_RUNTIME
        p_collector->start_timer("V-Second-Upload", trial_num, rep_num,
                                 cycle_num);
#endif
        // swap buffers
        std::swap(lookup_pointer, upload_pointer);

        // load all segments except last
        for (load_seg_idx = idle_segment_idx + 2;
             idle_segment_idx + load_seg_idx < num_segments_to_load;
             load_seg_idx++) {

            // ---upload data---
            awg->load_data_async_start(load_seg_idx, upload_pointer,
                                       samples_per_segment * sizeof(short));

            //---lookup waveforms for next segment
            wf_segment_lookup(lookup_pointer, moves_list,
                              waveforms_per_segment);

            // wait for old upload to finish
            awg->wait_for_data_load();

#ifdef LOGGING_RUNTIME
            if (load_seg_idx == idle_segment_idx + 2) {
                p_collector->end_timer("V-Second-Upload", trial_num, rep_num,
                                       cycle_num);
                p_collector->start_timer("V-First-Update", trial_num, rep_num,
                                         cycle_num);
            }
#endif

            // swap buffers
            std::swap(lookup_buffer, upload_buffer);

            // ---update sequence memory---
            old_control = idle_step_idx + load_seg_idx * 2 - 3;
            old_null = idle_step_idx + load_seg_idx * 2 - 2;
            new_control = idle_step_idx + load_seg_idx * 2 - 1;

            // point old control to new control
            awg->seqmem_update(old_control, load_seg_idx, 1, new_control,
                               SPCSEQ_ENDLOOPALWAYS);

            // point old null to new control
            awg->seqmem_update(old_null, null_segment_idx, 1, new_control,
                               SPCSEQ_ENDLOOPALWAYS);

            // if it's the second segment that just loaded, take playback off
            // idle
            if (load_seg_idx == (idle_segment_idx + 2)) {
                awg->seqmem_update(idle_segment_idx, idle_segment_idx, 1,
                                   old_control, SPCSEQ_ENDLOOPALWAYS);

#ifdef LOGGING_RUNTIME
                p_collector->end_timer("V-First-Update", trial_num, rep_num,
                                       cycle_num);
                p_collector->end_timer("V-Latency", trial_num, rep_num,
                                       cycle_num);
#endif
            }

            // reset idle as soon as we can
            if (!played_first_seg) {
                current_step = awg->get_current_step();
                if (current_step > idle_step_idx) {
                    INFO << "GOT OFF IDLE, load_seg_idx: " << load_seg_idx
                         << std::endl;
                    played_first_seg = 1;
                    awg->seqmem_update(idle_step_idx, idle_segment_idx, 1,
                                       idle_step_idx, SPCSEQ_ENDLOOPALWAYS);
                }
            }
        }

        // reset idle as soon as we can
        if (!played_first_seg) {
            current_step = awg->get_current_step();
            if (current_step > idle_step_idx) {
                played_first_seg = 1;
                awg->seqmem_update(idle_step_idx, idle_segment_idx, 1,
                                   idle_step_idx, SPCSEQ_ENDLOOPALWAYS);
            }
        }

        // --- All segments have been loaded past this point---
#ifdef LOGGING_VERBOSE
        INFO << "Loaded all " << num_segments_to_load << " segments"
             << std::endl;
        INFO << "Have we gone off idle? :" << played_first_seg << std::endl;
#endif

        // Query AWG step every 10 microseconds to see if we're finished
        // streaming
        bool streaming_too_long = false;
        auto beginning = HRCLOCK_NOW();
        auto start = HRCLOCK_NOW();
        while (!(current_step == idle_segment_idx && finished)) {

            auto now = HRCLOCK_NOW();

            if (microseconds_cast(now - start) > 10) {
                start = now;
                current_step = awg->get_current_step();

                // set "finished" flag if we have played the last segment
                // TODO: The system should ideally wait for a hardware trigger
                // to stop playing
                if (current_step == last_control_step) {
                    if (!finished) {
                        finished = 1;
                    }
                }

                // reset idle as soon as we can
                if (!played_first_seg) {
                    if (current_step > idle_segment_idx) {
                        played_first_seg = 1;
                        awg->seqmem_update(idle_segment_idx, idle_segment_idx,
                                           1, idle_segment_idx,
                                           SPCSEQ_ENDLOOPALWAYS);
                    }
                }
            }

#ifdef LOGGING_VERBOSE
            if (microseconds_cast(now - beginning) > (10 * num_moves) * 9 &&
                !streaming_too_long) {
                INFO << "Streaming time is 9 times longer than expected, "
                        "exitting"
                     << std::endl;
                INFO << "Current step = " << current_step << std::endl;
                INFO << "Last control step = " << last_control_step
                     << std::endl;
                INFO << "Short circuit seg idx = " << short_circuit_seg_idx
                     << std::endl;
                INFO << "Short circuit step = " << short_circuit_step
                     << std::endl;
                INFO << "Have we gone off idle? :" << played_first_seg
                     << std::endl;
                INFO << "Are we finished? :" << finished << std::endl;
                streaming_too_long = 1;
                break;
            }
#endif
        }

#ifdef LOGGING_RUNTIME
        p_collector->end_timer("V-Load_Stream", trial_num, rep_num, cycle_num);
#endif

        return streaming_too_long;
    }

    template <typename AWG_T>
    void Stream::Sequence<AWG_T>::wf_segment_lookup(
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

    template <typename AWG_T>
    void Stream::Sequence<AWG_T>::get_static_wfm(int16 * pnData,
                                                 size_t num_wfms, int Nt_x) {

        short *move_wf_ptr = wf_table.get_primary_wf_ptr(STATIC, 0, Nt_x, Nt_x);

        for (int i = 0; i < num_wfms; ++i) {
            memcpy(pnData + i * awg->get_waveform_length(), move_wf_ptr,
                   awg->get_waveform_length() * sizeof(short));
        }
    }

    template class Stream::Sequence<AWG>;
