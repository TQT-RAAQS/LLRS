#include <shot_file.h>
#include <movements_config.h>

int main() {
    std::string address = "/home/tqtraaqs1/Z/Experiments/Rydberg/2024-05-26/21_47_11-test_movement/labscript_shot_outputs/test_movement_2024-05-26_0021_0.h5";

    ShotFile shotfile(address);
    MovementsConfig movementsConfig(shotfile);

    std::vector<LabscriptDictType> movement_waveforms = movementsConfig.get_movement_waveforms();
    std::vector<long> movement_triggers = movementsConfig.get_movement_triggers();

    std::cout << boost::get<double>(movement_waveforms.at(1).at("duration")) * 1e6 << std::endl;
    std::cout << movement_triggers.at(0) << std::endl;
    
    return 0;
}