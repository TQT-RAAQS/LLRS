#ifndef _RAMSEY_STABILIZER_PID_LOOP_
#define _RAMSEY_STABILIZER_PID_LOOP_

#include "yaml-cpp/yaml.h"
#include <deque>

class PIDLoopController {

    YAML::Node configs;

    std::deque<double> buffer;
    size_t max_buffer_size;

    double k_p;
    double k_i;
    double k_d;
    double max_change;

    double get_p_correction();
    double get_i_correction();
    double get_d_correction();

public:

    PIDLoopController(YAML::Node configs);
    void set_control_param(double d);

    double compute_correction(double v);
};

#endif