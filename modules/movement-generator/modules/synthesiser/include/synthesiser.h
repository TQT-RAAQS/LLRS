#include "llrs-lib/Settings.h"
#include "awg.hpp"
#include "Waveform.h"
#include "Setup.h"
#include "globals-config.h"
#include <map>

enum WFFuncType {SIN_STATIC = 0,
                 TANH_TRANSITION = 1,
                 SPLINE_TRANSITION = 2,
                 ERF_TRANSITION = 3
                 };

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


class Synthesiser {
  public:
    struct Move {
        WfType type;
        int index, offset, block_size;
        double duration;
        WFFuncType func_type;
        double vmax;
        bool wait_for_trigger;
    };
    Synthesiser(std::string coef_x_path, std::string coef_y_path, MovementsConfig movementsConfig);
    void synthesise_and_upload(AWG& awg);
  private:
    std::vector<Synthesis::WP> coef_x, coef_y;
    std::vector<Move> moves;
	double sample_rate;
    Move process_move(MovementsConfig movementsConfig, int move_index); 
    std::vector<short> synthesise(Move move);
};
