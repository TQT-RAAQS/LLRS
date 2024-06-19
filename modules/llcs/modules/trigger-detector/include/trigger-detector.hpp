#ifndef TRIGGER_DETECTOR_HPP_
#define TRIGGER_DETECTOR_HPP_

#include "awg.hpp"
#include "llcs/common.hpp"
#include <cmath>
#include <fstream>
#include <memory>

template <typename AWG_T> class TriggerDetector {
    std::shared_ptr<AWG_T> awg;
    size_t samples_per_idle_segment;

  public:
    TriggerDetector();
    int stream();
    int setup(typename AWG_T::TransferBuffer &tb);
    int reset();
    int busyWait();
    int resetDetectionStep();
    int detectTrigger(int timeout = -1);
    std::shared_ptr<AWG_T> &getAWG() { return awg; }
    size_t get_samples_per_idle_segment() { return samples_per_idle_segment; }
};

#endif
