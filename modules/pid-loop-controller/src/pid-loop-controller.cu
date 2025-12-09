#include "pid-loop-controller.h"

PIDLoopController::PIDLoopController(YAML::Node configs) {
    this->configs = std::move(configs);
    this->param_max = this->configs["param_max"].as<double>();
    this->param_min = this->configs["param_max"].as<double>();
    this->max_buffer_size = this->configs["max_buffer_size"].as<size_t>();
    this->k_p = this->configs["k_p"].as<double>();
    this->k_i = this->configs["k_i"].as<double>();
    this->k_d = this->configs["k_d"].as<double>();
    this->set_control_param(this->configs["param_initial"].as<double>());
}

void PIDLoopController::set_control_param(double d) {
    this->param = d;
}

double PIDLoopController::get_control_param() {
    return this->param;
}

void PIDLoopController::add_value(double v) {
    this->buffer.push_back(v);
    if (this->buffer.size() > this->max_buffer_size) {
        this->buffer.pop_front();
    }

    double p_correction = this->get_p_correction();
    double i_correction = this->get_i_correction();
    double d_correction = this->get_d_correction();
    
    auto new_param = this->param + p_correction + i_correction + d_correction;

    this->param = max( min(new_param, this->param_max), this->param_min );
}

double PIDLoopController::get_p_correction() {
    if (this->buffer.size() < 2) {
        return 0.0;
    }

    double e_now = this->buffer.back();
    double e_prev = this->buffer[this->buffer.size() - 2];

    return this->k_p * (e_now - e_prev);
}

double PIDLoopController::get_i_correction() {
    return this->k_i * this->buffer.back();
}

double PIDLoopController::get_d_correction() {
    if (this->buffer.size() < 3) {
        return 0.0;
    }

    double e_now = this->buffer.back();
    double e_prev1 = this->buffer[this->buffer.size() - 2];
    double e_prev2 = this->buffer[this->buffer.size() - 3];

    return this->k_d * (e_now - 2*e_prev1 + e_prev2);
}