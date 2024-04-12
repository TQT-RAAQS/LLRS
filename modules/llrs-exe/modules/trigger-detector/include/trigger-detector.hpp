#ifndef TRIGGER_DETECTOR_HPP_
#define TRIGGER_DETECTOR_HPP_

#include "awg.hpp"
#include "llrs-exe/common.hpp"
#include <cmath>
#include <fstream>
#include <memory>

template <typename AWG_T> class TriggerDetector {
    std::shared_ptr<AWG_T> awg;
    int64_t samples_per_td_segment;

  public:
    TriggerDetector();
    void generateSineWave(int16 *pnData, int samples, double sampleRate);
    int setup(int16 *pnData);
    int resetDetectionSegments();
    int busyWait();
    int detectTrigger(int timeout = -1);
    std::shared_ptr<AWG_T> &getAWG();
};

#endif
