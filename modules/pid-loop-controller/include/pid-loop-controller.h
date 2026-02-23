#ifndef _RAMSEY_STABILIZER_PID_LOOP_
#define _RAMSEY_STABILIZER_PID_LOOP_

#include "yaml-cpp/yaml.h"
#include <deque>

class PIDLoopController {

    YAML::Node configs;

    size_t max_buffer_size;

    double max_change;

protected:

    std::deque<double> buffer;

    double k_p;
    double k_i;
    double k_d;

    int proportional_width;

    virtual double get_p_correction();
    virtual double get_i_correction();
    virtual double get_d_correction();

public:

    PIDLoopController(YAML::Node configs);

    double compute_correction(double v);
};

#endif