#ifndef _RAMSEY_STABILIZER_LINEAR_PID_LOOP_
#define _RAMSEY_STABILIZER_LINEAR_PID_LOOP_

#include "controller.h"
#include "yaml-cpp/yaml.h"
#include "configs-translator.h"
#include "llrs-lib/PreProc.h"
#include <fstream>
#include <deque>

class LinearController : public Controller {

    std::deque<double> error_buffer;
    std::deque<double> correction_buffer;

    double lp_alpha; // Low-pass filter alpha for the error signal; 1 means no filtering, 0 means infinite filtering.
    int64_t M; // How many shots before the current one to consider for the correction
    size_t max_buffer_size; // Maximum size of the error and correction buffers, should be at least M.
    bool force_reload;

    std::vector<double> error_coefficients; // Precomputed coefficients for the linear combination of past errors. Order is reverse-chronological.
    std::vector<double> correction_coefficients; // Precomputed coefficients for the linear combination of past corrections. Order is reverse-chronological.

    ConfigsTranslator& translator = ConfigsTranslator::instance();

    double calculate_correction_term();

public:

    LinearController(YAML::Node configs);

    double compute_correction(double error) override;
    
    void reset() override;
};

#endif