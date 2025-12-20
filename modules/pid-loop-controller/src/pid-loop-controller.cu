#include "pid-loop-controller.h"
#include <iostream>

PIDLoopController::PIDLoopController(YAML::Node configs) {
    this->configs = std::move(configs);
    this->max_buffer_size = this->configs["max_buffer_size"].as<size_t>();
    this->k_p = this->configs["k_p"].as<double>();
    this->k_i = this->configs["k_i"].as<double>();
    this->k_d = this->configs["k_d"].as<double>();
    this->max_change = this->configs["max_change"].as<double>();
}

double PIDLoopController::compute_correction(double v) {
    this->buffer.push_back(v);
    if (this->buffer.size() > this->max_buffer_size) {
        this->buffer.pop_front();
    }

    double p_correction = this->get_p_correction();
    double i_correction = this->get_i_correction();
    double d_correction = this->get_d_correction();
    
    auto raw_correction = p_correction + i_correction + d_correction;;

    return abs(raw_correction) > abs(max_change) ? raw_correction / abs(raw_correction) * abs(max_change) : raw_correction;
}

double PIDLoopController::get_p_correction() {
    if (this->buffer.size() < 2) {
        return 0.0;
    }

    double e_now = this->buffer.back();
    double e_prev = this->buffer[this->buffer.size() - 2];

    return -this->k_p * (e_now - e_prev);
}

double PIDLoopController::get_i_correction() {
    return -this->k_i * this->buffer.back();
}

double PIDLoopController::get_d_correction() {
    if (this->buffer.size() < 3) {
        return 0.0;
    }

    double e_now = this->buffer.back();
    double e_prev1 = this->buffer[this->buffer.size() - 2];
    double e_prev2 = this->buffer[this->buffer.size() - 3];

    return -this->k_d * (e_now - 2*e_prev1 + e_prev2);
}