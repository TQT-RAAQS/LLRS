#include "llrs-lib/Settings.h"
#include "awg.hpp"
#include "Waveform.h"
#include "Setup.h"
#include <map>

enum WFFuncType {SIN_STATIC, TANH_TRANSITION, SPLINE_TRANSITION, ERF_TRANSITION};


class Synthesiser {
  public:
    struct Move {
        WfType type;
        int index, offset, block_size;
        double duration;
        WFFuncType func_type;
        std::vector<double> func_params;
        bool wait_for_trigger;
    };
  private:
    std::vector<Synthesis::WP> coef_x, coef_y;
    std::vector<Move> moves;
    Move process_hashmap(std::map hashmap); 
    std::vector<short> synthesis(Move move);

  public:
    Synthesiser(std::string coef_x_path, std::string coef_y_path,std::vector<Move> moves);
    Synthesiser(std::string coef_x_path, std::string coef_y_path, std::vector<std::map> hashmaps);
    

    void synthesise_and_upload(AWG& awg);

};