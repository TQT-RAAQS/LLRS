#ifndef WAVEFORMPOWERCALCULATOR_H_
#define WAVEFORMPOWERCALCULATOR_H_

#include "llrs-lib/PreProc.h"
#include <cmath>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace Synthesis {

class WaveformPowerCalculator {

  public:
    WaveformPowerCalculator(int config_vpp);
    double get_power_mw(const std::vector<short> &waveform);
    double get_power_mw(const std::vector<double> &waveform);

    double get_power_dBm(const std::vector<short> &waveform);
    double get_power_dBm(const std::vector<double> &waveform);

    bool is_power_safe(const std::vector<short> &waveform);
    bool is_power_safe(const std::vector<double> &waveform);

  private:
    double mw_to_dBm(double power_mw);
    double dBm_to_mw(double power_dBm);

    double damage_threshold_dBm;
    double p0_mw;
    double vpp, vpp0;
    int config_vpp;
};
inline double WaveformPowerCalculator::mw_to_dBm(double power_mw) {
    return 10 * log10(power_mw);
}

inline double WaveformPowerCalculator::dBm_to_mw(double power_dBm) {
    return pow(10, power_dBm / 10);
}
} // namespace Synthesis

#endif //_WAVEFORMPOWERCALCULATOR_H_
