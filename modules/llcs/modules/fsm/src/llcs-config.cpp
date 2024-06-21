#include "llcs-config.h"

void LLCSConfig::translate_commands() {
    LLCSCommandType type;
    LLCSCommandData data;

    int iterator = -1;
    for (auto &cmd : *commands_int) {
        iterator++;
        type = static_cast<LLCSCommandType>(cmd);
        switch (type) {
            case MOVE_SHOT:
                commands.push_back(LLCSCommand(
                    LLCSCommandType::MOVE_SHOT,
                    get_move_command_data(iterator)
                ));
                break;

            case LLRS_SHOT:
                commands.push_back(LLCSCommand(
                    LLCSCommandType::LLRS_SHOT,
                    get_llrs_command_data(iterator)
                ));
                break;
        }
    }
}

MoveCommandData LLCSConfig::get_move_command_data(int index) {
    std::string key = "emccd_workstation_movement_waveforms_" + std::to_string(index);
    MoveCommandData result;
    shotfile.get_global_list_dict(key, &result);

    return  result;
}

LLRSCommandData LLCSConfig::get_llrs_command_data(int index) {
    std::string key = "emccd_workstation_llrs_" + std::to_string(index);
    LLRSCommandData result;
    shotfile.get_global_dict(key, &result);

    return result;
}