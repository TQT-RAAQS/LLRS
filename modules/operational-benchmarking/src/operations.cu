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

    std::vector<int32_t> target_config(Nt_x * Nt_y);
    create_rectangular_target(target_config, Nt_x * Nt_y, num_target, Nt_x,
                              Nt_y);

    Reconfig::Solver solver;
    solver.setup(Nt_x, Nt_y, 32);
    std::vector<uint> alpha_ops;
    std::vector<uint> nu_ops;
    alpha_ops.reserve(num_reps * num_trials);
    nu_ops.reserve(num_reps * num_trials);
    // Start trial loop
    for (int trial = 0; trial < num_trials; ++trial) {
        // Create initial atom configuration with 60% loading efficiency
        std::vector<int32_t> trial_config(Nt_x * Nt_y);
        double loading_efficiency = 0.6;
        for (auto &it : trial_config) {
            it = (loading_efficiency >= (((double)rand()) / RAND_MAX)) ? 1 : 0;
        }
        if (Util::count_num_atoms(trial_config) >= num_target) {
            // Start repetition loop
            for (int rep = 0; rep < num_reps; ++rep) {
                solver.start_solver(algo, trial_config, target_config);
                auto moves_list = solver.gen_moves_list(algo, batching);
                uint alpha_op = 0;
                uint nu_op = 0;
                for (auto &move : moves_list) {
                    if (std::get<0>(move) == Synthesis::IMPLANT_2D || std::get<0>(move) == Synthesis::EXTRACT_2D) {
                        ++alpha_op;
                    } else if (std::get<0>(move) != Synthesis::IDLE_1D, std::get<0>(move) != Synthesis::IDLE_2D) {
                        ++nu_op;
                    }
                }
                alpha_ops.push_back(alpha_op);
                nu_ops.push_back(nu_op);
            }
        }
    }

    if (alpha_ops.size() == 0) {
        std::cout << "0, 0" << std::endl << "0, 0" << std::endl;
    } else {
        {
            double mean = std::accumulate(alpha_ops.begin(), alpha_ops.end(), 0.0) / alpha_ops.size();
            std::vector<double> diffs;
            diffs.reserve(alpha_ops.size());
            for (auto it : alpha_ops) {
                diffs.push_back(it - mean);
            }
            double stddev = std::sqrt(
            std::inner_product(diffs.begin(), diffs.end(), diffs.begin(), 0.0) /
            (alpha_ops.size() - 1));
            std::cout << mean << ", " << stddev << std::endl;
        }
        {
            double mean = std::accumulate(nu_ops.begin(), nu_ops.end(), 0.0) / nu_ops.size();
            std::vector<double> diffs;
            diffs.reserve(nu_ops.size());
            for (auto it : nu_ops) {
                diffs.push_back(it - mean);
            }
            double stddev = std::sqrt(
            std::inner_product(diffs.begin(), diffs.end(), diffs.begin(), 0.0) /
            (nu_ops.size() - 1));
            std::cout << mean << ", " << stddev << std::endl;
 
        }
    }

    
    return 0;
}
