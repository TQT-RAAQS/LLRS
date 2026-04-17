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