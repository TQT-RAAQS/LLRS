#ifndef RAMSEY_STABILIZER_LABSCRIPT_CONFIG_H_
#define RAMSEY_STABILIZER_LABSCRIPT_CONFIG_H_

#include "globals-config.h"

class RamseyStabilizerLabscriptConfig : protected GlobalsConfig {
    
    int ramsey_stabilizer_active_flag;
    double ramsey_stabilizer_k_p;
    double ramsey_stabilizer_k_i;
    double ramsey_stabilizer_k_d;
    double ramsey_stabilizer_k_p_width;
    double ramsey_stabilizer_first_gate_phase;
    double ramsey_stabilizer_nu0;
    double ramsey_stabilizer_alpha;
    double ramsey_stabilizer_phi0;
    double ramsey_stabilizer_pi2_T;
    double ramsey_stabilizer_second_gate_phase;
    double ramsey_stabilizer_tau;
    double ramsey_stabilizer_max_change;
    int ramsey_stabilizer_gradient_x_parallel;
    int ramsey_stabilizer_clear_memory_flag;
    int ramsey_stabilizer_pid_enabled;
    int ramsey_stabilizer_active_pid_index;
    int controller_type;
    int ramsey_stabilizer_nu_buffer_size;
    int ramsey_stabilizer_track_mode;
    double ramsey_stabilizer_track_mode_factor;
    char* mw_signals;
    bool qdac_controller_active;

  public:
  RamseyStabilizerLabscriptConfig(const std::string& shot_address)
        : GlobalsConfig(
              shot_address,
              {{"ramsey_stabilizer_k_p", &ramsey_stabilizer_k_p, LabscriptType::VALUE},
               {"ramsey_stabilizer_k_i", &ramsey_stabilizer_k_i, LabscriptType::VALUE},
               {"ramsey_stabilizer_k_d", &ramsey_stabilizer_k_d, LabscriptType::VALUE},
               {"ramsey_stabilizer_first_gate_phase", &ramsey_stabilizer_first_gate_phase, LabscriptType::VALUE},
               {"ramsey_stabilizer_nu0", &ramsey_stabilizer_nu0, LabscriptType::VALUE},
               {"ramsey_stabilizer_alpha", &ramsey_stabilizer_alpha, LabscriptType::VALUE},
               {"ramsey_stabilizer_phi0", &ramsey_stabilizer_phi0, LabscriptType::VALUE},
               {"ramsey_stabilizer_pi2_T", &ramsey_stabilizer_pi2_T, LabscriptType::VALUE},
               {"ramsey_stabilizer_second_gate_phase", &ramsey_stabilizer_second_gate_phase, LabscriptType::VALUE},
               {"ramsey_stabilizer_tau", &ramsey_stabilizer_tau, LabscriptType::VALUE},
               {"mw_signals", &mw_signals, LabscriptType::VALUE},
               {"ramsey_stabilizer_max_change", &ramsey_stabilizer_max_change, LabscriptType::VALUE},
               {"ramsey_stabilizer_active_flag", &ramsey_stabilizer_active_flag, LabscriptType::VALUE},
               {"ramsey_stabilizer_pid_enabled", &ramsey_stabilizer_pid_enabled, LabscriptType::VALUE},
               {"ramsey_stabilizer_active_pid_index", &ramsey_stabilizer_active_pid_index, LabscriptType::VALUE},
               {"ramsey_stabilizer_clear_memory_flag", &ramsey_stabilizer_clear_memory_flag, LabscriptType::VALUE},
               {"ramsey_stabilizer_k_p_width", &ramsey_stabilizer_k_p_width, LabscriptType::VALUE},
               {"ramsey_stabilizer_controller_type", &controller_type, LabscriptType::VALUE},
               {"ramsey_stabilizer_nu_buffer_size", &ramsey_stabilizer_nu_buffer_size, LabscriptType::VALUE},
               {"ramsey_stabilizer_track_mode", &ramsey_stabilizer_track_mode, LabscriptType::VALUE},
               {"ramsey_stabilizer_track_mode_factor", &ramsey_stabilizer_track_mode_factor, LabscriptType::VALUE},
               {"qdac_controller_active", &qdac_controller_active, LabscriptType::VALUE},
               {"ramsey_stabilizer_flag_g_x_parallel", &ramsey_stabilizer_gradient_x_parallel, LabscriptType::VALUE}}) {}

    int get_ramsey_stabilizer_active_flag() const { return ramsey_stabilizer_active_flag; }
    double get_ramsey_stabilizer_k_p() const { return ramsey_stabilizer_k_p; }
    double get_ramsey_stabilizer_k_i() const { return ramsey_stabilizer_k_i; }
    double get_ramsey_stabilizer_k_d() const { return ramsey_stabilizer_k_d; }
    double get_ramsey_stabilizer_k_p_width() const { return ramsey_stabilizer_k_p_width; }
    double get_ramsey_stabilizer_first_gate_phase() const { return ramsey_stabilizer_first_gate_phase; }
    double get_ramsey_stabilizer_nu0() const { return ramsey_stabilizer_nu0; }
    double get_ramsey_stabilizer_alpha() const { return ramsey_stabilizer_alpha; }
    double get_ramsey_stabilizer_phi0() const { return ramsey_stabilizer_phi0; }
    double get_ramsey_stabilizer_pi2_T() const { return ramsey_stabilizer_pi2_T; }
    double get_ramsey_stabilizer_second_gate_phase() const { return ramsey_stabilizer_second_gate_phase; }
    double get_ramsey_stabilizer_tau() const { return ramsey_stabilizer_tau; }
    double get_ramsey_stabilizer_max_change() const { return ramsey_stabilizer_max_change; }
    double get_ramsey_stabilizer_track_mode_factor() const { return ramsey_stabilizer_track_mode_factor; }
    int get_ramsey_stabilizer_active_pid_index() const { return ramsey_stabilizer_active_pid_index; }
    int get_ramsey_stabilizer_gradient_x_parallel() const { return ramsey_stabilizer_gradient_x_parallel; }
    int get_ramsey_stabilizer_pid_enabled() const { return ramsey_stabilizer_pid_enabled; }
    int get_controller_type() const { return controller_type; }
    int get_ramsey_stabilizer_nu_buffer_size() const { return ramsey_stabilizer_nu_buffer_size; }
    bool get_ramsey_stabilizer_clear_memory_flag() const { return ramsey_stabilizer_clear_memory_flag != 0; }
    bool get_ramsey_stabilizer_track_mode() const { return ramsey_stabilizer_track_mode != 0; }
    bool get_qdac_controller_active() const { return qdac_controller_active; }
    std::string get_mw_signals() const { return mw_signals; }
};

#endif