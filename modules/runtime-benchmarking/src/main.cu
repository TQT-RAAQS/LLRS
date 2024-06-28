#include "ImageProcessor-replace.h"
#include "acquisition-stored.h"
#include "llrs.h"
#include <cstdlib>
#include <fstream>
#include <jsoncpp/json/json.h>
#include <memory>

using CYCLE_INFO = std::vector<std::vector<std::tuple<std::string, long long>>>;
using REP_INFO = std::vector<CYCLE_INFO>;
using TRIAL_INFO = std::vector<REP_INFO>;

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
    std::shared_ptr<Processing::ImageProcessorReplace> image_processor{
        std::make_shared<Processing::ImageProcessorReplace>()};
    LLRS l{
        awg,
        std::dynamic_pointer_cast<Acquisition::ImageAcquisition>(
            std::make_shared<Acquisition::ImageAcquisitionStored>()),
        std::dynamic_pointer_cast<Processing::ImageProcessor>(image_processor)};
    l.setup(problem_config, true, 0);

    awg->start_stream();
    awg->print_awg_error();
    assert(awg->get_current_step() == 0);
    TRIAL_INFO timers;
    Json::Value problem_soln = Util::read_json_file(SOLN_PATH(problem_id));
    for (size_t trial_num = 0; problem_soln.isMember(TRIAL_NAME(trial_num));
         ++trial_num) {
        Json::Value trial_soln = problem_soln[TRIAL_NAME(trial_num)];
        REP_INFO trial_timers;
        /* SWAP with pre-solved from processed */
        for (int rep_num = 0; trial_soln.isMember(REP_NAME(rep_num));
             ++rep_num) {
            Json::Value rep_soln = trial_soln[REP_NAME(rep_num)];
            std::vector<std::vector<int32_t>> configs;
            for (int cycle_num = 0; rep_soln.isMember(CYCLE_NAME(cycle_num));
                 ++cycle_num) {
                configs.push_back(
                    Util::vector_transform(rep_soln[CYCLE_NAME(cycle_num)]));
            }
            image_processor->set_configs(configs);
            l.execute();
            trial_timers.push_back(l.getMetadata().getRuntimeData());
        }
        l.reset(true);
        timers.push_back(trial_timers);
    }

    Json::Value timing_data;
    for (int i = 0; i < timers.size(); ++i) {
        std::string trial_name = TRIAL_NAME(i);
        const REP_INFO &rep_info = timers[i];
        for (int j = 0; j < rep_info.size(); ++j) {
            std::string repetition_name = REP_NAME(j);
            const CYCLE_INFO &cycle_info = rep_info[j];
            for (int k = 0; k < cycle_info.size(); k++) {
                std::string cycle_name = CYCLE_NAME(k);
                for (int l = 0; l < cycle_info[k].size(); l++) {
                    std::string module = std::get<0>(cycle_info[k][l]);
                    timing_data[trial_name][repetition_name][cycle_name]
                               [module] = std::get<1>(cycle_info[k][l]);
                }
            }
        }
    }
    std::string output_fname = BENCHMARK_PATH(problem_id);
    Util::write_json_file(timing_data, output_fname);
    return LLRS_OK;
}
