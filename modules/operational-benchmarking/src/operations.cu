#include "Solver.h"
#include "llrs-lib/Settings.h"
#include "trap-array.h"
#include "utility.h"
#include <chrono>
#include <cstring>
#include <random>
#include <string>

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
    if (argc < 8) {
        std::cout << "Usage is operational_benchmarking <algorithm> <Nt_x> "
                     "<Nt_y> <num_target> "
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
    std::vector<std::vector<uint>> alpha_ops;
    std::vector<std::vector<uint>> nu_ops;
    std::vector<std::vector<uint>> EDIs;
    alpha_ops.reserve(num_reps * num_trials);
    nu_ops.reserve(num_reps * num_trials);
    EDIs.reserve(num_reps * num_trials);
    // Start trial loop
    for (int trial = 0; trial < num_trials; ++trial) {

        // Create initial atom configuration with 60% loading efficiency
        std::vector<int32_t> trial_config(Nt_x * Nt_y);
        {
            double loading_efficiency = 0.6;
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<std::mt19937::result_type> dist6(
                0, RAND_MAX);
            for (auto &it : trial_config) {
                it = (loading_efficiency >= (((double)dist6(rng)) / RAND_MAX))
                         ? 1
                         : 0;
            }
        }
        if (Util::count_num_atoms(trial_config) >= num_target) {

            // Start repetition loop
            for (int rep = 0; rep < num_reps; ++rep) {
                std::vector<int32_t> rep_config{trial_config};
                std::vector<uint> cycle_alpha;
                std::vector<uint> cycle_nu;
                std::vector<uint> cycle_EDI;
                int cycle = 0;
                while (!Util::target_met(rep_config, target_config) &&
                       cycle < 9) {

                    // Initialize trap array object
                    TrapArray trap_array(Nt_x, Nt_y, rep_config, 0.985, 0.985,
                                         60);

                    // check if we meet the required target config
                    // if we do not, then check if we have enough atoms to
                    // continue the process

                    if (Util::count_num_atoms(rep_config) < num_target) {
                        break;
                    }

                    std::vector<Reconfig::Move> moves_list;
                    solver.start_solver(algo, rep_config, target_config);
                    moves_list = solver.gen_moves_list(algo, batching);
                    uint alpha_op = 0;
                    uint nu_op = 0;
                    uint EDI = 0;
                    bool Extracted = false;
                    for (auto &move : moves_list) {
                        if (std::get<0>(move) == Synthesis::IMPLANT_2D ||
                            std::get<0>(move) == Synthesis::EXTRACT_2D) {
                            ++alpha_op;
                        } else if (std::get<0>(move) != Synthesis::IDLE_1D,
                                   std::get<0>(move) != Synthesis::IDLE_2D) {
                            ++nu_op;
                        }
                        if (std::get<0>(move) == Synthesis::EXTRACT_2D) {
                            Extracted = true;
                        } else if (std::get<0>(move) == Synthesis::IMPLANT_2D &&
                                   Extracted) {
                            ++EDI;
                            Extracted = false;
                        } else if (std::get<0>(move) == Synthesis::IMPLANT_2D) {
                            throw std::runtime_error("Implant without Extract");
                        }
                    }
                    cycle_alpha.push_back(alpha_op);
                    cycle_nu.push_back(nu_op);
                    cycle_EDI.push_back(EDI);
                    // Performs the moves
                    if (trap_array.performMoves(moves_list) != 0) {
                        break;
                    }

                    // Performs loss
                    trap_array.performLoss();
                    trap_array.getArray(rep_config);
                    solver.reset();
                    ++cycle;
                }
                alpha_ops.push_back(cycle_alpha);
                nu_ops.push_back(cycle_nu);
                EDIs.push_back(cycle_EDI);
            }
        } else {
            --trial;
        }
    }

    if (alpha_ops.size() == 0) {
        std::cout << "Not successful." << std::endl;
    } else {
        for (auto it : alpha_ops) {
            std::cout << it.at(0) << ", ";
        }
        std::cout << std::endl;
        for (auto it : nu_ops) {
            std::cout << it.at(0) << ", ";
        }
        std::cout << std::endl;
        for (auto it : EDIs) {
            std::cout << it.at(0) << ", ";
        }
        std::cout << std::endl;

        std::vector<std::vector<double>> alpha_ops_cycled{15};
        for (auto it : alpha_ops) {
            for (size_t i = 0; i < it.size(); ++i) {
                alpha_ops_cycled[i].push_back(it[i]);
            }
        }
        std::vector<std::vector<double>> nu_ops_cycled{15};
        for (auto it : nu_ops) {
            for (size_t i = 0; i < it.size(); ++i) {
                nu_ops_cycled[i].push_back(it[i]);
            }
        }
        std::vector<std::vector<double>> edis_cycled{15};
        for (auto it : EDIs) {
            for (size_t i = 0; i < it.size(); ++i) {
                edis_cycled[i].push_back(it[i]);
            }
        }
        for (auto it : alpha_ops_cycled) {
            if (it.size() == 0) {
                continue;
            }
            double mean_alpha =
                std::accumulate(it.begin(), it.end(), 0.0) / it.size();
            std::vector<double> diffs_alpha;
            diffs_alpha.reserve(it.size());
            for (auto it2 : it) {
                diffs_alpha.push_back(it2 - mean_alpha);
            }
            if (it.size() == 1) {
                std::cout << mean_alpha << ", 0 - ";
                continue;
            }
            double stddev_alpha = std::sqrt(
                std::inner_product(diffs_alpha.begin(), diffs_alpha.end(),
                                   diffs_alpha.begin(), 0.0) /
                (it.size() - 1));
            std::cout << mean_alpha << ", " << stddev_alpha << " - ";
        }
        std::cout << std::endl;

        for (auto it : nu_ops_cycled) {
            if (it.size() == 0) {
                continue;
            }
            double mean_alpha =
                std::accumulate(it.begin(), it.end(), 0.0) / it.size();
            std::vector<double> diffs_alpha;
            diffs_alpha.reserve(it.size());
            for (auto it2 : it) {
                diffs_alpha.push_back(it2 - mean_alpha);
            }
            if (it.size() == 1) {
                std::cout << mean_alpha << ", 0 - ";
                continue;
            }
            double stddev_alpha = std::sqrt(
                std::inner_product(diffs_alpha.begin(), diffs_alpha.end(),
                                   diffs_alpha.begin(), 0.0) /
                (it.size() - 1));
            std::cout << mean_alpha << ", " << stddev_alpha << " - ";
        }
        std::cout << std::endl;

        for (auto it : edis_cycled) {
            if (it.size() == 0) {
                continue;
            }
            double mean_alpha =
                std::accumulate(it.begin(), it.end(), 0.0) / it.size();
            std::vector<double> diffs_alpha;
            diffs_alpha.reserve(it.size());
            for (auto it2 : it) {
                diffs_alpha.push_back(it2 - mean_alpha);
            }
            if (it.size() == 1) {
                std::cout << mean_alpha << ", 0 - ";
                continue;
            }
            double stddev_alpha = std::sqrt(
                std::inner_product(diffs_alpha.begin(), diffs_alpha.end(),
                                   diffs_alpha.begin(), 0.0) /
                (it.size() - 1));
            std::cout << mean_alpha << ", " << stddev_alpha << " - ";
        }
    }
    return 0;
}
