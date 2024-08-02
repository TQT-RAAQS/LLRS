#include "Collector.h"
#include "Solver.h"
#include "llrs-lib/Settings.h"
#include "utility.h"
#include <cstring>
#include <string>
#include <chrono>
#include <random>

/**
 * @brief Creates target in the center of the trap array, filling the width of
 * the array
 */

void create_rectangular_target(std::vector<int32_t> &target_config, int num_target, int Nt_x,
                               int Nt_y) {

    int target_height = num_target / Nt_x;

    int start_row = (Nt_y - target_height) / 2;
    int end_row = start_row + target_height;

    int start_index = start_row * Nt_x;

    for (int i = 0; i < num_target; i++) {
        target_config[start_index + i] = 1;
    }
}

/**
 * @brief Operational benchmarking for the reconfiguration algorithm
 *
 * @param argc Number of arguments: 3
 * @param argv Arguments: problem_file_path, num_trials, num_repititions
 * @return int
 */
int main(int argc, char *argv[]) {
    if (argc != 8) {
        std::cout << "Usage is operational_benchmarking <algorithm> <Nt_x> <Nt_y> <num_target> "
                     "<num_trials> <num_repititions> <batching>"
                  << std::endl;
        return 1;
    }

    std::string algorithm{argv[1]};
    int Nt_x = std::stoi(argv[2]);
    int Nt_y = std::stoi(argv[3]);
    int num_target = std::stoi(argv[4]);
    int num_trials = std::stoi(argv[5]);
    int num_reps = std::stoi(argv[6]);
    bool batching = std::stoi(argv[7]);
    Reconfig::Algo algo{Util::get_algo_enum(algorithm)};

    std::vector<int32_t> target_config(Nt_x * Nt_y, 0);
    create_rectangular_target(target_config, num_target, Nt_x,
                              Nt_y);

    Reconfig::Solver solver;
    solver.setup(Nt_x, Nt_y, 32);
    std::vector<double> data;
    data.reserve(num_trials * num_reps);
    // Start trial loop
    for (int trial = 0; trial < num_trials; ++trial) {
        std::vector<int32_t> trial_config(Nt_x * Nt_y, 0);
        for (size_t i = 0; i < num_target; i++) {
            trial_config[i] = 1;
        }
        std::random_device rd;
        std::mt19937 g(rd());

        // Shuffle the vector
        std::random_shuffle(trial_config.begin(), trial_config.end(), [&](int n) {
            return g() % n;
        }); 
        // Start repetition loop
        for (int rep = 0; rep < num_reps; ++rep) {
            solver.start_solver(algo, trial_config, target_config);
            if (batching) {
                solver.gen_moves_list(algo, true);
            }
            data.push_back(batching ? Util::Collector::get_instance()->get_module("III-Matching") + Util::Collector::get_instance()->get_module("III-Batching"):Util::Collector::get_instance()->get_module("III-Matching"));
            Util::Collector::get_instance()->clear_timers();
            solver.reset();
        }
    }
    if (data.size() == 0) {
        std::cout << "0, 0" << std::endl;
    } else {
        double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        std::vector<double> diffs;
        diffs.reserve(data.size());
        for (auto it : data) {
            diffs.push_back(it - mean);
        }
        double stddev = std::sqrt(
        std::inner_product(diffs.begin(), diffs.end(), diffs.begin(), 0.0) /
        (data.size() - 1));
        std::cout << mean << ", " << stddev << std::endl;
    }

    
    return 0;
}
