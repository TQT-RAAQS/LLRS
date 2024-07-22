#include "Solver.h"
#include "llrs-lib/Settings.h"
#include "trap-array.h"
#include "utility.h"
#include <cstring>
#include <string>

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
    if (argc != 12) {
        std::cout << "Usage is operational_benchmarking "
                     "<algorithm> <Ntx> <Nty> <num_target> <loading_efficiency> "
                     "<alpha> <nu> <lifetime> <num_trials> <num_repititions> <batched>"
                  << std::endl;
        return 1;
    }

    std::string algorithm = argv[1];
    Reconfig::Algo algo{Util::get_algo_enum(algorithm)};
    int Nt_x = std::stoi(argv[2]);
    int Nt_y = std::stoi(argv[3]);
    int num_target = std::stoi(argv[4]);
    double loading_efficiency = std::stod(argv[5]);
    double loss_params_alpha = std::stod(argv[6]);
    double loss_params_nu = std::stod(argv[7]);
    double lifetime = std::stod(argv[8]);
    int num_trials = std::stoi(argv[9]);
    int num_reps = std::stoi(argv[10]);
    bool batched = std::stoi(argv[11]);

    std::vector<int32_t> target_config(Nt_x * Nt_y);
    create_rectangular_target(target_config, Nt_x * Nt_y, num_target, Nt_x,
                              Nt_y);
    int required_num = num_target;

    int successes = 0;
    bool failure = false;

    // Start trial loop
    for (int trial = 0; trial < num_trials; ++trial) {
        failure = false;

        // Create initial atom configuration with 60% loading efficiency
        std::vector<int32_t> trial_config(Nt_x * Nt_y);
        for (auto &it : trial_config) {
            it = (loading_efficiency >= (((double)rand()) / RAND_MAX)) ? 1 : 0;
        }

        // Start repetition loop
        for (int rep = 0; rep < num_reps; ++rep) {

            failure = false;

            std::vector<int32_t> rep_config(trial_config);

            int cycle = 0;
            int numSuccess = 0;
            bool not_enough_atoms = false;

            while (!Util::target_met(rep_config, target_config)) {

                Reconfig::Solver solver;
                solver.setup(Nt_x, Nt_y, 32);

                // Initialize trap array object
                TrapArray trap_array (Nt_x, Nt_y, rep_config, loss_params_alpha,
                              loss_params_nu, lifetime);

                failure = false;
                // check if we meet the required target config
                // if we do not, then check if we have enough atoms to continue
                // the process

                if (Util::count_num_atoms(rep_config) < required_num) {
                    not_enough_atoms = true;
                    break;
                }

                // Start solver and generate moves list

                std::vector<Reconfig::Move> moves_list;
                solver.start_solver(algo, rep_config, target_config);
                moves_list = solver.gen_moves_list(algo, batched);

                // Performs the moves
                if (trap_array.performMoves(moves_list) != 0) {
                    failure = true;
                    break;
                }

                // Performs loss
                trap_array.performLoss();
                trap_array.getArray(rep_config);

                cycle++;
            }

            if (not_enough_atoms) {
                continue;
            }
            if (failure) {
                continue;
            }
            ++successes;
        }
        if (failure) {
            continue;
        }
    }

    std::cout << (double)((double)successes / (double)(num_reps * num_trials))
              << std::endl;
    return 0;
}
