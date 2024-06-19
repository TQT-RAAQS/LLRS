#include "WaveformPowerCalculator.h"

Synthesis::WaveformPowerCalculator::WaveformPowerCalculator(int config_vpp) {
    try {
        YAML::Node config =
            YAML::LoadFile(POWER_SAFETY_CONFIG_PATH("config.yml"));
        this->on = config["on"].as<bool>();
        this->damage_threshold_dBm = config["danger_threshold"].as<double>();
        this->p0_mw = config["mono_p_max"].as<double>();
        this->vpp0 = config["vpp"].as<double>();
        this->config_vpp = config_vpp;
    } catch (...) {
        std::cerr << "Error loading power safety config." << std::endl;
        throw;
    }
}

double Synthesis::WaveformPowerCalculator::get_power_mw(
    const std::vector<short> &waveform) {
    double avg_power = 0;
    const double factor = 2 * this->p0_mw / waveform.size() / 268435456 *
                          config_vpp * config_vpp / this->vpp0 / this->vpp0;

    for (const double &sample : waveform) {
        avg_power += factor * sample * sample;
    }
    return avg_power;
}
double Synthesis::WaveformPowerCalculator::get_power_mw(
    const std::vector<double> &waveform) {
    double avg_power = 0;
    const double factor = 2 * this->p0_mw / waveform.size() * config_vpp *
                          config_vpp / this->vpp0 / this->vpp0;
    ;

    for (const double &sample : waveform) {
        avg_power += factor * sample * sample;
    }
    return avg_power;
}

double Synthesis::WaveformPowerCalculator::get_power_dBm(
    const std::vector<short> &waveform) {
    const double power_mw = this->get_power_mw(waveform);
    return this->mw_to_dBm(power_mw);
}
double Synthesis::WaveformPowerCalculator::get_power_dBm(
    const std::vector<double> &waveform) {
    const double power_mw = this->get_power_mw(waveform);
    return this->mw_to_dBm(power_mw);
}

bool Synthesis::WaveformPowerCalculator::is_power_safe(
    const std::vector<short> &waveform) {
    return this->get_power_dBm(waveform) <= this->damage_threshold_dBm;
}
bool Synthesis::WaveformPowerCalculator::is_power_safe(
    const std::vector<double> &waveform) {
    return this->get_power_dBm(waveform) <= this->damage_threshold_dBm;
}
