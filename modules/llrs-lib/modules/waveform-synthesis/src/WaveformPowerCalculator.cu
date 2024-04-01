#include "WaveformPowerCalculator.h"

Synthesis::WaveformPowerCalculator::WaveformPowerCalculator() {
    // This has to be read from a config file before being merged.
    this->damage_threshold_dBm = 32;
    this->p0_mw = 4466.835922;
    this->vpp0 = 280;
}

double Synthesis::WaveformPowerCalculator::get_power_mw(
    const std::vector<short> &waveform) {
    double avg_power = 0;
    const double factor = 2 * this->p0_mw / waveform.size() / 268435456 * VPP *
                          VPP / this->vpp0 / this->vpp0;

    for (const double &sample : waveform) {
        avg_power += factor * sample * sample;
    }
    return avg_power;
}
double Synthesis::WaveformPowerCalculator::get_power_mw(
    const std::vector<double> &waveform) {
    double avg_power = 0;
    const double factor =
        2 * this->p0_mw / waveform.size() * VPP * VPP / this->vpp0 / this->vpp0;
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
