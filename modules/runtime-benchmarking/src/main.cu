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
    LLRS l{awg};
    l.setup(problem_config, true, 0, problem_id);

    awg->start_stream();
    awg->print_awg_error();
    assert(awg->get_current_step() == 0);

    Json::Value problem_soln = Util::read_json_file(SOLN_PATH(problem_id));
    for (trial_num = 0; problem_soln.isMember(TRIAL_NAME(trial_num));
         trial_num++) {
        Json::Value trial_soln = problem_soln[TRIAL_NAME(trial_num)];
        for (rep_num = 0; trial_soln.isMember(REP_NAME(rep_num)); rep_num++) {
            Json::Value rep_soln = trial_soln[REP_NAME(rep_num)];
            l.execute();
            l.reset(true);   
        }
    }
}
