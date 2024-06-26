#ifndef TRIGGER_DETECTOR_HPP_
#define TRIGGER_DETECTOR_HPP_

#include "awg.hpp"
#include "llcs/common.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <memory>

class TriggerDetector {
    std::shared_ptr<AWG> awg;
    size_t samples_per_idle_segment;

  public:
    TriggerDetector();
    int stream();
    int setup(typename AWG::TransferBuffer &tb);
    int reset();
    int busyWait();
    int resetDetectionStep();
    int detectTrigger(int timeout = -1);
    std::shared_ptr<AWG> &getAWG() { return awg; }
    size_t get_samples_per_idle_segment() { return samples_per_idle_segment; }
    void reset_segment_size() {
        samples_per_idle_segment = awg->get_idle_segment_length();
    }
};

#endif
