#ifndef HDF5_TEST_CONFIG_H_
#define HDF5_TEST_CONFIG_H_

#include <globals_config.h>

class MovementsConfig : protected GlobalsConfig {
    bool enabled;
    int movement_count;
    std::vector<long>* movement_triggers;
    std::vector<LabscriptDictType>* movement_waveforms;

public:
    MovementsConfig(ShotFile shotfile) : GlobalsConfig(shotfile, {
        {"movement_enabled", &enabled, LabscriptType::VALUE},
        {"movement_count", &movement_count, LabscriptType::VALUE},
        {"movement_triggers", &movement_triggers, LabscriptType::LIST_OF_LONG},
        {"movement_waveforms", &movement_waveforms, LabscriptType::LIST_OF_DICT}
    }) {}

    bool is_enabled() const { return enabled; }
    int get_movement_count() const { return movement_count; }
    std::vector<LabscriptDictType> get_movement_waveforms() const { return *movement_waveforms; }
    std::vector<long> get_movement_triggers() const { return *movement_triggers; }
};

#endif