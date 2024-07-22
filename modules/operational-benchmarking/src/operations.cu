#include "Solver.h"
#include "llrs-lib/Settings.h"
#include "utility.h"
#include <cstring>
#include <string>
#include <chrono>
#include <random>

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
    if (argc != 9) {
        std::cout << "Usage is operational_benchmarking <algorithm> <Nt_x> <Nt_y> <num_target> "
                     "<num_trials> <num_repititions> <batching> <raw>"
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
    bool raw = std::stoi(argv[8]);
    Reconfig::Algo algo{Util::get_algo_enum(algorithm)};

    std::vector<int32_t> target_config(Nt_x * Nt_y);
    create_rectangular_target(target_config, Nt_x * Nt_y, num_target, Nt_x,
                              Nt_y);


    std::vector<uint> alpha_ops;
    std::vector<uint> nu_ops;
    std::vector<uint> EDIs;
    std::vector<uint> displacements;
    alpha_ops.reserve(num_reps * num_trials);
    nu_ops.reserve(num_reps * num_trials);
    EDIs.reserve(num_reps * num_trials);
    displacements.reserve(num_reps * num_trials);
    // Start trial loop
    for (int trial = 0; trial < num_trials; ++trial) {
        // Create initial atom configuration with 60% loading efficiency
        std::vector<int32_t> trial_config(Nt_x * Nt_y);
        double loading_efficiency = 0.6;
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist6(0, RAND_MAX);
        Reconfig::Solver solver;
        solver.setup(Nt_x, Nt_y, 32);
        for (auto &it : trial_config) {
            it = (loading_efficiency >= (((double)dist6(rng)) / RAND_MAX)) ? 1 : 0;
        } 
        if (Util::count_num_atoms(trial_config) >= num_target) {
            // Start repetition loop
            for (int rep = 0; rep < num_reps; ++rep) {
                solver.start_solver(algo, trial_config, target_config);
                auto moves_list = solver.gen_moves_list(algo, batching);
                uint alpha_op = 0;
                uint nu_op = 0;
                uint EDI = 0;
                uint displacement = 0;
                bool Extracted = false;
                for (auto &move : moves_list) {
                    if (std::get<0>(move) == Synthesis::IMPLANT_2D || std::get<0>(move) == Synthesis::EXTRACT_2D) {
                        ++alpha_op;
                    } else if (std::get<0>(move) != Synthesis::IDLE_1D, std::get<0>(move) != Synthesis::IDLE_2D) {
                        ++nu_op;
                        displacement += std::get<3>(move);
                    }
                    if (std::get<0>(move) == Synthesis::EXTRACT_2D) {
                        Extracted = true;
                    } else if (std::get<0>(move) == Synthesis::IMPLANT_2D && Extracted) {
                        ++EDI;
                        Extracted = false;
                    } else if (std::get<0>(move) == Synthesis::IMPLANT_2D) {
                        throw std::runtime_error("Implant without Extract");
                    }
                }
                alpha_ops.push_back(alpha_op);
                nu_ops.push_back(nu_op);
                EDIs.push_back(EDI);
                displacements.push_back(displacement);
            }
        }
    }

    if (alpha_ops.size() == 0) {
        std::cout << "0, 0" << std::endl << "0, 0" << "0, 0" << std::endl;
    } else {
        if (raw) {
            for (auto it : alpha_ops) {
                std::cout << it << ", ";
            }
            std::cout << std::endl;
            for (auto it : nu_ops) {
                std::cout << it << ", ";
            }
            std::cout << std::endl;
            for (auto it : EDIs) {
                std::cout << it << ", ";
            }
            std::cout << std::endl; 
            for (auto it : displacements) {
                std::cout << it << ", ";
            }
            std::cout << std::endl;
            return;
        }
        double mean_alpha = std::accumulate(alpha_ops.begin(), alpha_ops.end(), 0.0) / alpha_ops.size();
        std::vector<double> diffs_alpha;
        diffs_alpha.reserve(alpha_ops.size());
        for (auto it : alpha_ops) {
            diffs_alpha.push_back(it - mean_alpha);
        }
        double stddev_alpha = std::sqrt(
        std::inner_product(diffs_alpha.begin(), diffs_alpha.end(), diffs_alpha.begin(), 0.0) /
        (alpha_ops.size() - 1));
        
        double mean_nu = std::accumulate(nu_ops.begin(), nu_ops.end(), 0.0) / nu_ops.size();
        std::vector<double> diffs_nu;
        diffs_nu.reserve(nu_ops.size());
        for (auto it : nu_ops) {
            diffs_nu.push_back(it - mean_nu);
        }
        double stddev_nu = std::sqrt(
        std::inner_product(diffs_nu.begin(), diffs_nu.end(), diffs_nu.begin(), 0.0) /
        (nu_ops.size() - 1));

        double mean_edi = std::accumulate(EDIs.begin(), EDIs.end(), 0.0) / EDIs.size();
        std::vector<double> diffs_edi;
        diffs_edi.reserve(EDIs.size());
        for (auto it : EDIs) {
            diffs_edi.push_back(it - mean_edi);
        }
        double stddev_edi = std::sqrt(
        std::inner_product(diffs_edi.begin(), diffs_edi.end(), diffs_edi.begin(), 0.0) /
        (EDIs.size() - 1));

        double mean_displacement = std::accumulate(displacements.begin(), displacements.end(), 0.0) / displacements.size();
        std::vector<double> diffs_displacement;
        diffs_displacement.reserve(EDIs.size());
        for (auto it : displacements) {
            diffs_displacement.push_back(it - mean_displacement);
        }
        double stddev_displacement = std::sqrt(
        std::inner_product(diffs_displacement.begin(), diffs_displacement.end(), diffs_displacement.begin(), 0.0) /
        (displacements.size() - 1));

 
        std::cout << mean_alpha << ", " << stddev_alpha << std::endl;
        std::cout << mean_nu << ", " << stddev_nu << std::endl;
        std::cout << mean_edi << ", " << stddev_edi << std::endl;
        std::cout << mean_displacement << ", " << stddev_displacement << std::endl;
    }
    
    return 0;
}
