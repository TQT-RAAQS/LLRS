#include "shot-file.h"
#include "globals-config.h"

using LLRSCommandData = LabscriptDictType;
using MoveCommandData = std::vector<LabscriptDictType>;
using LLCSCommandData = boost::variant<LLRSCommandData, MoveCommandData>;

enum LLCSCommandType {
    MOVE_SHOT = 0,
    LLRS_SHOT = 1
};


class LLCSCommand {

  public:
    const LLCSCommandType type;
    const LLCSCommandData data;

    LLCSCommand(LLCSCommandType type, LLCSCommandData data) :
        type(type), data(data) {}
};

class LLCSConfig : protected GlobalsConfig {

    std::vector<int> *commands_int;
    std::vector<LLCSCommand> commands;

    void translate_commands();
    MoveCommandData get_move_command_data(int index);
    LLRSCommandData get_llrs_command_data(int index);

  public:
    
    LLCSConfig(ShotFile shotfile)
        : GlobalsConfig(
              shotfile,
              {{"emccd_workstation_commands", &commands_int, LabscriptType::LIST_OF_INT},
               }) {
                    translate_commands();
               }


    std::vector<LLCSCommand> get_commands() { return commands; }
};