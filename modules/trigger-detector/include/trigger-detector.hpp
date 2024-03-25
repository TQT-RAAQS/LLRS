#ifndef TRIGGER_DETECTOR_HPP
#define TRIGGER_DETECTOR_HPP

#include "llcs/common.hpp"
#include "awg.hpp"
#include <memory>
#include <cmath>
#include <fstream>


template<typename AWG_T> class TriggerDetector{
    std::shared_ptr<AWG_T> awg;
    int64_t samples_per_td_segment;

public:
    TriggerDetector();
    void generateSineWave( int16 *pnData, int samples, double sampleRate );
    int setup( int16 *pnData );
    int resetDetectionSegments();
    int busyWait( float timeout = -1 );
    int detectTrigger( int timeout = -1 );
    std::shared_ptr<AWG_T> &getAWG();
};

#endif
