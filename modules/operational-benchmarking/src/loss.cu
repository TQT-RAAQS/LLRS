#include "Solver.h"
#include "llrs-lib/Settings.h"
#include "trap-array.h"
#include "utility.h"
#include <cstring>
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
    if (argc != 9) {
        std::cout << "Usage is operational_benchmarking "
                     "<algorithm> <Ntx> <Nty> <num_target> <loading_efficiency> "
                     "<alpha> <nu> <num_repititions>"
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
    int num_reps = std::stoi(argv[8]);
    std::vector<int32_t> target_config(Nt_x * Nt_y);
    create_rectangular_target(target_config, Nt_x * Nt_y, num_target, Nt_x,
                              Nt_y);
    int required_num = num_target;

    bool failure = false;

    // Create initial atom configuration with 60% loading efficiency
    std::vector<int32_t> trial_config(Nt_x * Nt_y);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist6(0, RAND_MAX);

    for (auto &it : trial_config) {
        it = (loading_efficiency >= (((double)dist6(rng)) / RAND_MAX)) ? 1 : 0;
    }
    uint num_atom = Util::count_num_atoms(trial_config);
    std::vector<uint> lifetimes {1,3,6,9,12,30,60,90,120,300,600, 900, 1200};

    for(auto lifetime: lifetimes) {
        std::vector<uint> num_losts;
        num_losts.reserve(num_reps);
        // Start repetition loop
        for (int rep = 0; rep < num_reps; ++rep) {
            std::vector<int32_t> rep_config(trial_config);
            failure = false;
            int cycle = 0;
            int numSuccess = 0;
            bool not_enough_atoms = false;
            size_t num_lost = 0;
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
                moves_list = solver.gen_moves_list(algo, false);

                // Performs the moves
                if (trap_array.performMoves(moves_list) != 0) {
                    failure = true;
                    break;
                }

                // Performs loss
                num_lost += trap_array.performLoss();
                trap_array.getArray(rep_config);

                cycle++;
            }
            num_losts.push_back(num_lost);
            if (not_enough_atoms) {
                continue;
            }
            if (failure) {
                continue;
            }
        }

        {
            std::vector<double> data;
            data.reserve(num_reps);
            for (size_t i = 0; i < num_reps; i++) {
                data.push_back((double)num_losts[i] / num_atom);
            }
            double mean = (double)std::accumulate(data.begin(), data.end(), 0.0) / data.size();
            std::vector<double> diffs;
            diffs.reserve(data.size());
            for (auto it : data) {
                diffs.push_back(it - mean);
            }
            double stddev = std::sqrt(
            std::inner_product(diffs.begin(), diffs.end(), diffs.begin(), 0.0) /
            (data.size() - 1));
            std::cout << mean << ", " << stddev;
        }
        std::cout << " - ";
        {
            double mean = std::accumulate(num_losts.begin(), num_losts.end(), 0.0) / num_losts.size();
            std::vector<double> diffs;
            diffs.reserve(num_losts.size());
            for (auto it : num_losts) {
                diffs.push_back(it - mean);
            }
            double stddev = std::sqrt(
            std::inner_product(diffs.begin(), diffs.end(), diffs.begin(), 0.0) /
            (num_losts.size() - 1));
            std::cout << mean << ", " << stddev;
        }
        std::cout << " | ";
    }
    std::cout << std::endl;
    for(auto lifetime: lifetimes) {
        std::vector<uint> num_losts;
        num_losts.reserve(num_reps);
        // Start repetition loop
        for (int rep = 0; rep < num_reps; ++rep) {
            std::vector<int32_t> rep_config(trial_config);
            failure = false;
            int cycle = 0;
            int numSuccess = 0;
            bool not_enough_atoms = false;
            size_t num_lost = 0;
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
                moves_list = solver.gen_moves_list(algo, true);

                // Performs the moves
                if (trap_array.performMoves(moves_list) != 0) {
                    failure = true;
                    break;
                }

                // Performs loss
                num_lost += trap_array.performLoss();
                trap_array.getArray(rep_config);

                cycle++;
            }
            num_losts.push_back(num_lost);
            if (not_enough_atoms) {
                continue;
            }
            if (failure) {
                continue;
            }
        }

        {
            std::vector<double> data;
            data.reserve(num_reps);
            for (size_t i = 0; i < num_reps; i++) {
                data.push_back((double)num_losts[i] / num_atom);
            }
            double mean = (double)std::accumulate(data.begin(), data.end(), 0.0) / data.size();
            std::vector<double> diffs;
            diffs.reserve(data.size());
            for (auto it : data) {
                diffs.push_back(it - mean);
            }
            double stddev = std::sqrt(
            std::inner_product(diffs.begin(), diffs.end(), diffs.begin(), 0.0) /
            (data.size() - 1));
            std::cout << mean << ", " << stddev;
        }
        std::cout << " - ";
        {
            double mean = std::accumulate(num_losts.begin(), num_losts.end(), 0.0) / num_losts.size();
            std::vector<double> diffs;
            diffs.reserve(num_losts.size());
            for (auto it : num_losts) {
                diffs.push_back(it - mean);
            }
            double stddev = std::sqrt(
            std::inner_product(diffs.begin(), diffs.end(), diffs.begin(), 0.0) /
            (num_losts.size() - 1));
            std::cout << mean << ", " << stddev;
        }
        std::cout << " | ";
    }
    std::cout << std::endl;




    return 0;
}
