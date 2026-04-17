#ifndef RAMSEY_STABILIZER_60HZ_MODEL_H
#define RAMSEY_STABILIZER_60HZ_MODEL_H

#include "configs-translator.h"
#include "llrs-lib/PreProc.h"

struct RamseyStabilizer60HzModelParameters {

    double b, A, T_A, B, T_B, a1, a2, dt, nu_AC;

    // Equation: A exp(-t/T_A) + B exp(-t/T_B) + a1 cos(2 pi nu_AC (t+dt)) + a2 cos(2 pi 3 nu_AC (t+dt)) + b
    // Integral: A T_A (1 - exp(-t/T_A)) + B T_B (1 - exp(-t/T_B)) + a1/(2 pi nu_AC) sin(2 pi nu_AC (t+dt)) + a2/(2 pi 3 nu_AC) sin(2 pi 3 nu_AC (t+dt)) + b t
};

class RamseyStabilizer60HzModel {

    ConfigsTranslator& config_translator = ConfigsTranslator::instance();
    RamseyStabilizer60HzModelParameters parameters{};

    public:
        RamseyStabilizer60HzModel();

        void reload_parameters();
        inline double get_phase_correction(double t) const {
            return this->parameters.A * (1 - std::exp(-t / this->parameters.T_A)) + 
                   this->parameters.B * (1 - std::exp(-t / this->parameters.T_B)) + 
                   this->parameters.a1 / (2 * M_PI * this->parameters.nu_AC) * std::sin(2 * M_PI * this->parameters.nu_AC * (t + this->parameters.dt)) + 
                   this->parameters.a2 / (2 * M_PI * 3 * this->parameters.nu_AC) * std::sin(2 * M_PI * 3 * this->parameters.nu_AC * (t + this->parameters.dt)) + 
                   this->parameters.b * t;
        }
};

#endif