#ifndef RAMSEY_STABILIZER_PHASE_PID_LOOP_
#define RAMSEY_STABILIZER_PHASE_PID_LOOP_

#include "pid-loop-controller.h"
#include "yaml-cpp/yaml.h"
#include <deque>

class PIDLoopPhaseController : public PIDLoopController {

    static double wrap_phase(double phase);

protected:
    double get_p_correction() override;
    double get_d_correction() override;

public:

    PIDLoopPhaseController(YAML::Node configs) : PIDLoopController(configs) {}

};

#endif