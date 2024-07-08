#include "Collector.h"
#include "Solver.h"
#include "llrs-lib/Settings.h"
#include "utility.h"
#include <cstring>
#include <string>
#include <chrono>

void create_center_target(std::vector<int32_t> &target_config, int num_trap,
                          int num_target) {

    int start_index = (num_trap / 2) - (num_target / 2);

    for (int offset = 0; offset < num_target; ++offset) {
        target_config[start_index + offset] = 1;
    }
}
std::vector<int32_t> get_target_config(std::string target, int num_trap,
                                       int num_target) {
    std::vector<int32_t> target_config(num_trap, 0);

    if (target == "center compact") {
        create_center_target(target_config, num_trap, num_target);
        return target_config;
    } else {
        std::cerr << "ERROR: Desired configuration not available" << std::endl;
    }
}

/**
 * @brief Creates target in the center of the trap array, filling the width of
 * the array
 */

void create_rectangular_target(std::vector<int32_t> &target_config,
                               int num_trap, int num_target, int Nt_x,
                               int Nt_y) {

    int target_width = num_target / Nt_x;

    int start_row = (Nt_y - target_width) / 2;
    int end_row = start_row + num_trap / Nt_x;

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
    if (argc != 7) {
        std::cout << "Usage is operational_benchmarking <algorithm> <Nt_x> <Nt_y> <num_target> "
                     "<num_trials> <num_repititions>"
                  << std::endl;
        return 1;
    }

    std::string algorithm{argv[1]};
    int Nt_x = std::stoi(argv[2]);
    int Nt_y = std::stoi(argv[3]);
    int num_target = std::stoi(argv[4]);
    int num_trials = std::stoi(argv[5]);
    int num_reps = std::stoi(argv[6]);
    Reconfig::Algo algo{Util::get_algo_enum(algorithm)};

    std::vector<int32_t> target_config(Nt_x * Nt_y);
    create_rectangular_target(target_config, Nt_x * Nt_y, num_target, Nt_x,
                              Nt_y);

    Reconfig::Solver solver;
    solver.setup(Nt_x, Nt_y, 32);
    double data = 0;
    int success_trials = 0;
    // Start trial loop
    for (int trial = 0; trial < num_trials; ++trial) {
        // Create initial atom configuration with 60% loading efficiency
        std::vector<int32_t> trial_config(Nt_x * Nt_y);
        double loading_efficiency = 0.6;
        for (auto &it : trial_config) {
            it = (loading_efficiency >= (((double)rand()) / RAND_MAX)) ? 1 : 0;
        }
        if (Util::count_num_atoms(trial_config) >= num_target) {
            ++success_trials;
            double trial_data = 0; 
            // Start repetition loop
            for (int rep = 0; rep < num_reps; ++rep) {
                solver.start_solver(algo, trial_config, target_config);
                trial_data += Util::Collector::get_instance()->get_module("III-Matching");
                Util::Collector::get_instance()->clear_timers();
            }
            data += (trial_data / num_reps);
        }
    }
    if (success_trials == 0) {
        std::cout << 0 << std::endl;
    } else {
        std::cout << data / num_trials << std::endl;
    }

    
    return 0;
}
