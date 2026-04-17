#include "ramsey-stabilizer-60hz-model.h"

RamseyStabilizer60HzModel::RamseyStabilizer60HzModel() {
    this->reload_parameters();
}

void RamseyStabilizer60HzModel::reload_parameters() {
    this->config_translator.translate_ramsey_stabilizer_60hz_model();

    std::ifstream infile(RAMSEY_STABILIZER_60HZ_MODEL_FILE, std::ios::binary);
    
    infile.read(reinterpret_cast<char*>(&this->parameters.b), sizeof(double));
    infile.read(reinterpret_cast<char*>(&this->parameters.A), sizeof(double));
    infile.read(reinterpret_cast<char*>(&this->parameters.T_A), sizeof(double));
    infile.read(reinterpret_cast<char*>(&this->parameters.B), sizeof(double));
    infile.read(reinterpret_cast<char*>(&this->parameters.T_B), sizeof(double));
    infile.read(reinterpret_cast<char*>(&this->parameters.a1), sizeof(double));
    infile.read(reinterpret_cast<char*>(&this->parameters.a2), sizeof(double));
    infile.read(reinterpret_cast<char*>(&this->parameters.dt), sizeof(double));
    infile.read(reinterpret_cast<char*>(&this->parameters.nu_AC), sizeof(double));

    infile.close();
}

// Integral: A T_A (1 - exp(-t/T_A)) + B T_B (1 - exp(-t/T_B)) + a1/(2 pi nu_AC) sin(2 pi nu_AC (t+dt)) + a2/(2 pi 3 nu_AC) sin(2 pi 3 nu_AC (t+dt)) + b t
double RamseyStabilizer60HzModel::get_phase_correction(double t) const {
    return this->parameters.A * (1 - std::exp(-t / this->parameters.T_A)) + 
           this->parameters.B * (1 - std::exp(-t / this->parameters.T_B)) + 
           this->parameters.a1 / (2 * M_PI * this->parameters.nu_AC) * std::sin(2 * M_PI * this->parameters.nu_AC * (t + this->parameters.dt)) + 
           this->parameters.a2 / (2 * M_PI * 3 * this->parameters.nu_AC) * std::sin(2 * M_PI * 3 * this->parameters.nu_AC * (t + this->parameters.dt)) + 
           this->parameters.b * t;
}