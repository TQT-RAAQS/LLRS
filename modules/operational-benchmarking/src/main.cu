#include "JsonWrapper.h"
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
    if (argc != 4) {
        std::cout << "Usage is operational_benchmarking <problem_file_path> "
                     "<num_trials> <num_repititions> "
                  << std::endl;
        return 1;
    }

    std::string file_path{argv[1]};
    int num_trials = std::stoi(argv[2]);
    int num_reps = std::stoi(argv[3]);

    // Now read file and convert to JSON object
    Util::JsonWrapper json_file(file_path);
    std::string algorithm = json_file.read_problem_algo();
    Reconfig::Algo algo{Util::get_algo_enum(algorithm)};

    // Read in initial config, target config from JSON object and problem
    // regions
    int Nt_x = json_file.read_problem_Nt_x();
    int Nt_y = json_file.read_problem_Nt_y();
    const int num_target = json_file.read_problem_num_target();
    std::vector<int32_t> target_config(Nt_x * Nt_y);

    // creates a symmetrical rectangular target configuration, filling the width
    // of the trap array

    create_rectangular_target(target_config, Nt_x * Nt_y, num_target, Nt_x,
                              Nt_y);

    // Read configurations from JSON file
    int required_num = Util::count_num_atoms(target_config);
    double loss_params_alpha = json_file.read_problem_alpha();
    double loss_params_nu = json_file.read_problem_nu();
    double lifetime = json_file.read_problem_lifetime();

    // Number of implantation or extraction moves, used for debugging the
    // batching
    int implantation_extraction_cnt = 0;

    int successes = 0;

    bool failure = false;

    // Start trial loop
    for (int trial = 0; trial < num_trials; ++trial) {
        failure = false;

        // Create initial atom configuration with 60% loading efficiency
        std::vector<int32_t> trial_config(Nt_x * Nt_y);

        for (auto &it : trial_config) {
            it = (0.6 >= (((double)rand()) / RAND_MAX)) ? 1 : 0;
        }

        // Start repetition loop
        for (int rep = 0; rep < num_reps; ++rep) {

            failure = false;

            std::vector<int32_t> rep_config(trial_config);

            int cycle = 0;
            int numSuccess = 0;
            bool not_enough_atoms = false;

            while (!Util::target_met(rep_config, target_config)) {

                Reconfig::Solver solver(Nt_x, Nt_y, nullptr);

                // Initialize trap array object
                TrapArray trap_array =
                    TrapArray(Nt_x, Nt_y, rep_config, loss_params_alpha,
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
                solver.start_solver(algo, rep_config, target_config, 0, 0, 0);
                moves_list = solver.gen_moves_list(algo, 0, 0, 0);

// Output the moves list and count number of extraction/implantation
#ifdef DEBUG
                for (auto i : moves_list) {
                    std::cout << std::get<0>(i) << " x " << std::get<1>(i)
                              << " y " << std::get<2>(i) << " b "
                              << std::get<3>(i) << std::endl;
                    if (std::get<0>(i) == 6 || std::get<0>(i) == 7) {
                        implantation_extraction_cnt++;
                    }
                }
#endif

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
