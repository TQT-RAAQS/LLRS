#include "pid-loop-phase-controller.h"

double PIDLoopPhaseController::wrap_phase(double phase) {
    // Wrap phase to the range [-pi, pi]
    auto p = std::fmod(phase + M_PI, 2.0 * M_PI);
    return (p > 0 ? p - M_PI : p + M_PI);
}

double PIDLoopPhaseController::get_p_correction() {
    if (this->buffer.size() < 2) {
        return 0.0;
    }

    double e_now = this->buffer.back();
    double e_prev = this->buffer[this->buffer.size() - 2];

    return -this->k_p * PIDLoopPhaseController::wrap_phase(e_now - e_prev);
}

double PIDLoopPhaseController::get_d_correction() {
    if (this->buffer.size() < 3) {
        return 0.0;
    }

    double e_now = this->buffer.back();
    double e_prev1 = this->buffer[this->buffer.size() - 2];
    double e_prev2 = this->buffer[this->buffer.size() - 3];

    return -this->k_d * PIDLoopPhaseController::wrap_phase(e_now - 2*e_prev1 + e_prev2);
}