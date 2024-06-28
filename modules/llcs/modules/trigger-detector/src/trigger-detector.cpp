#include "trigger-detector.hpp"
/**
 * @brief Class that contains the necessary code to detect hardware trigger
 * events
 */
TriggerDetector::TriggerDetector()
    : awg{std::make_shared<AWG>()}, samples_per_idle_segment{
                                        awg->get_idle_segment_length()} {}

int TriggerDetector::setup(typename AWG::TransferBuffer &tb) {

    int status = 0;

    // DATA MEMORY
    status |= awg->init_and_load_range(*tb, samples_per_idle_segment, 0, 1);

    // SEQUENCE MEMORY
    status |= awg->seqmem_update(0, 0, 1, 1, SPCSEQ_ENDLOOPONTRIG);
    status |= awg->seqmem_update(1, 0, 1, 1, SPCSEQ_ENDLOOPALWAYS);

    // Ensure there is enough time for the first idle segment's pointer to
    // update
    status |= busyWait();

    return status;
}

int TriggerDetector::stream() { return awg->start_stream(); }

int TriggerDetector::reset() {
    assert(awg->get_current_step() == 1);
    awg->seqmem_update(0, 0, 1, 1, SPCSEQ_ENDLOOPONTRIG);
    awg->seqmem_update(1, 0, 1, 0, SPCSEQ_ENDLOOPALWAYS);
    while (awg->get_current_step() != 0) {
    }
    awg->seqmem_update(1, 0, 1, 1, SPCSEQ_ENDLOOPALWAYS);
}

int TriggerDetector::busyWait() {
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

int TriggerDetector::resetDetectionStep() {
    int status;
    status |= awg->seqmem_update(0, 0, 1, 1, SPCSEQ_ENDLOOPONTRIG);
    status |= awg->seqmem_update(1, 0, 1, 1, SPCSEQ_ENDLOOPALWAYS);
    return status;
}

/**
 * @brief Function that detects a hardware trigger
 *
 * If a timeout value is given, then that value is used, otherwise
 * it will listen for a hw trigger forever.
 */
int TriggerDetector::detectTrigger(int timeout) {
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
