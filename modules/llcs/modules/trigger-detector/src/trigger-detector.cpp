#include "trigger-detector.hpp"
/**
 * @brief Class that contains the necessary code to detect hardware trigger
 * events
 */
template <typename AWG_T>
TriggerDetector<AWG_T>::TriggerDetector()
    : awg{std::make_shared<AWG_T>()}, samples_per_idle_segment{
                                          awg->get_idle_segment_length()} {
    int status = 0;

    // DATA MEMORY
    // trigger-detector initilizes and loads the first segment
    auto tb = awg->allocate_transfer_buffer(samples_per_idle_segment, false);
    awg->fill_transfer_buffer(tb, samples_per_idle_segment, 0);
    status |= setup(tb);

    if (!status) {
        std::cerr << "Problem while trigger detector's setup." << std::endl;
    }
}

template <typename AWG_T>
int TriggerDetector<AWG_T>::setup(typename AWG_T::TransferBuffer &tb) {

    int status = 0;

    // DATA MEMORY
    status |= awg->init_and_load_range(*tb, samples_per_idle_segment, 0, 1);

    // SEQUENCE MEMORY
    status |= awg->seqmem_update(0, 0, 1, 1, SPCSEQ_ENDLOOPONTRIG);

    // Ensure there is enough time for the first idle segment's pointer to
    // update
    status |= busyWait();

    return status;
}

template <typename AWG_T> int TriggerDetector<AWG_T>::stream() {
    return awg->start_stream();
}

template <typename AWG_T> int TriggerDetector<AWG_T>::busyWait() {
    float timeout =
        awg->get_waveform_duration() * awg->get_idle_segment_length() * 1e6;
    auto startTime = std::chrono::high_resolution_clock::now();
    auto targetDuration = std::chrono::duration<float, std::micro>(timeout);

    while (true) {

        // Timeout loop break
        if (timeout != -1 &&
            std::chrono::high_resolution_clock::now() - startTime >=
                targetDuration) {
            break;
        }
    }

    return 0;
}

template <typename AWG_T> int TriggerDetector<AWG_T>::resetDetectionStep() {

    return awg->seqmem_update(0, 0, 1, 1, SPCSEQ_ENDLOOPONTRIG);
}

/**
 * @brief Function that detects a hardware trigger
 *
 * If a timeout value is given, then that value is used, otherwise
 * it will listen for a hw trigger forever.
 */
template <typename AWG_T>
int TriggerDetector<AWG_T>::detectTrigger(int timeout) {
    int current_step = awg->get_current_step();
    int last_step = current_step;

    auto startTime = std::chrono::high_resolution_clock::now();
    auto targetDuration = std::chrono::seconds(timeout);

    while (true) {
        // Timeout loop break
        if (timeout != -1 &&
            std::chrono::high_resolution_clock::now() - startTime >=
                targetDuration) {
            break;
        }

        current_step = awg->get_current_step();
        if ((current_step != last_step)) {
            // trigger event occured
            std::cout
                << "TriggerDetector: trigger event occured - Current Step: "
                << current_step << std::endl;
            std::cout << "last_step: " << last_step << std::endl;
            last_step = current_step;

            return current_step;
        }
    }

    return -1;
}

template class TriggerDetector<AWG>;
