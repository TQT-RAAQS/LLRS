#include "Setup.h"
#include "Waveform.h"
#include "awg.hpp"
#include "globals-config.h"
#include "llrs-lib/Settings.h"
#include <unordered_map>

enum WFFuncType {
    SIN_STATIC = 0,
    TANH_TRANSITION = 1,
    SPLINE_TRANSITION = 2,
    ERF_TRANSITION = 3
};

class MovementsConfig : protected GlobalsConfig {
    bool enabled;
    int movement_count;
    std::vector<LabscriptDictType> *movement_waveforms;

  public:
    MovementsConfig(ShotFile shotfile)
        : GlobalsConfig(
              shotfile,
              {{"movement_enabled", &enabled, LabscriptType::VALUE},
               {"movement_count", &movement_count, LabscriptType::VALUE},
               {"movement_waveforms", &movement_waveforms,
                LabscriptType::LIST_OF_DICT}}) {}

    bool is_enabled() const { return enabled; }
    int get_movement_count() const { return movement_count; }
    std::vector<LabscriptDictType> get_movement_waveforms() const {
        return *movement_waveforms;
    }
};

class Synthesiser {
  public:
    struct Move {
        WfType type;
        int index, offset, block_size, extraction_extent;
        double duration;
        WFFuncType func_type;
        double vmax = 0;
        bool wait_for_trigger;
        bool operator==(const Move &other) const;
    };
    struct MoveHasher {
        size_t operator()(const Move &move) const;
    };
    Synthesiser(std::string coef_x_path, std::string coef_y_path);
    void synthesise_and_upload(AWG &awg, int start_segment);
    void reset(AWG &awg, int start_segment);
    void set_config(MovementsConfig movementsConfig);

  private:
    std::vector<Synthesis::WP> coef_x, coef_y;
    std::vector<Move> moves;
    double sample_rate;
    Move process_move(MovementsConfig &movementsConfig, int move_index);
    std::vector<short> synthesise(Move move);
    static std::unordered_map<Move, std::vector<short>, Synthesiser::MoveHasher>
        cache;
};
