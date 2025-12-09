#ifndef _RAMSEY_STABILIZER_LABSCRIPT_CONFIG_H_
#define _RAMSEY_STABILIZER_LABSCRIPT_CONFIG_H_

#include "globals-config.h"

class RamseyStabilizerLabscriptConfig : protected GlobalsConfig {
    
    double ramsey_stabilizer_delta_max;
    double ramsey_stabilizer_delta_min;
    double ramsey_stabilizer_k_p;
    double ramsey_stabilizer_k_i;
    double ramsey_stabilizer_k_d;
    double ramsey_stabilizer_first_gate_phase;
    double ramsey_stabilizer_nu0;
    double ramsey_stabilizer_phi0;
    double ramsey_stabilizer_pi2_T;
    double ramsey_stabilizer_second_gate_phase;
    double ramsey_stabilizer_tau;

  public:
  RamseyStabilizerLabscriptConfig(ShotFile shotfile)
        : GlobalsConfig(
              shotfile,
              {{"ramsey_stabilizer_delta_max", &ramsey_stabilizer_delta_max, LabscriptType::VALUE},
               {"ramsey_stabilizer_delta_min", &ramsey_stabilizer_delta_min, LabscriptType::VALUE},
               {"ramsey_stabilizer_k_p", &ramsey_stabilizer_k_p, LabscriptType::VALUE},
               {"ramsey_stabilizer_k_i", &ramsey_stabilizer_k_i, LabscriptType::VALUE},
               {"ramsey_stabilizer_k_d", &ramsey_stabilizer_k_d, LabscriptType::VALUE},
               {"ramsey_stabilizer_first_gate_phase", &ramsey_stabilizer_first_gate_phase, LabscriptType::VALUE},
               {"ramsey_stabilizer_nu0", &ramsey_stabilizer_nu0, LabscriptType::VALUE},
               {"ramsey_stabilizer_phi0", &ramsey_stabilizer_phi0, LabscriptType::VALUE},
               {"ramsey_stabilizer_pi2_T", &ramsey_stabilizer_pi2_T, LabscriptType::VALUE},
               {"ramsey_stabilizer_second_gate_phase", &ramsey_stabilizer_second_gate_phase, LabscriptType::VALUE},
               {"ramsey_stabilizer_tau", &ramsey_stabilizer_tau, LabscriptType::VALUE}}) {}

    double get_ramsey_stabilizer_delta_max() const { return ramsey_stabilizer_delta_max; }
    double get_ramsey_stabilizer_delta_min() const { return ramsey_stabilizer_delta_min; }
    double get_ramsey_stabilizer_k_p() const { return ramsey_stabilizer_k_p; }
    double get_ramsey_stabilizer_k_i() const { return ramsey_stabilizer_k_i; }
    double get_ramsey_stabilizer_k_d() const { return ramsey_stabilizer_k_d; }
    double get_ramsey_stabilizer_first_gate_phase() const { return ramsey_stabilizer_first_gate_phase; }
    double get_ramsey_stabilizer_nu0() const { return ramsey_stabilizer_nu0; }
    double get_ramsey_stabilizer_phi0() const { return ramsey_stabilizer_phi0; }
    double get_ramsey_stabilizer_pi2_T() const { return ramsey_stabilizer_pi2_T; }
    double get_ramsey_stabilizer_second_gate_phase() const { return ramsey_stabilizer_second_gate_phase; }
    double get_ramsey_stabilizer_tau() const { return ramsey_stabilizer_tau; }
};

#endif