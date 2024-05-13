#include "awg.hpp"
#include "llrs.h"
#include <cstdlib>
#include <fstream>
#include <memory>

int main(int argc, char *argv[]) {

    // Read problem statement
    std::string problem_id;
    std::string problem_config;
    if (argc > 2) {
        problem_config = std::string(argv[1]);
        problem_id = std::string(argv[2]);
    } else {
        std::cout << "For runtime benchmarking, please provide problem_id and "
                     "problem_config."
                  << std::endl;
        return LLRS_ERR;
    }

    std::shared_ptr<AWG> awg{std::make_shared<AWG>()};
    LLRS<AWG> l{awg};
    l.setup(problem_config, true, 0, problem_id);

    awg->start_stream();
    awg->print_awg_error();
    assert(awg->get_current_step() == 0);

    l.execute();
    l.reset();
}
