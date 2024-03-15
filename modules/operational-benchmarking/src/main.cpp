#include "trap-array.h"
#include "Solver.h"
#include "JsonWrapper.h"
#include "utility.h"
#include <cstring>
#include <string>
#include "llrs-lib/Settings.h"

void create_center_target( std::vector<int32_t> &target_config, int num_trap, int num_target ){
    
    int start_index = (num_trap / 2) - (num_target / 2);

    for (int offset = 0; offset < num_target; ++offset) {
        target_config[start_index + offset] = 1;
    }
    
}
std::vector<int32_t> get_target_config( std::string target, int num_trap, int num_target){
    std::vector<int32_t> target_config(num_trap, 0);

    if(target == "center compact"){
        create_center_target( target_config, num_trap, num_target );
        return target_config;
    } else {
        std::cerr << "ERROR: Desired configuration not available" << std::endl;
    }
}


/**
 * @brief Operational benchmarking for the reconfiguration algorithm
 * 
 * @param argc Number of arguments: 4  
 * @param argv Arguments: problem_file_path, num_trials, num_repititions, algorithm 
 * @return int 
 */
int main(int argc, char * argv[]) {
    if (argc != 5) {
        std::cout << "Usage is operational_benchmarking <problem_file_path> <num_trials> <num_repititions> <algorithm>" << std::endl;
        return 1;
    }

    std::string file_path {argv[1]};
    int num_trials = std::stoi(argv[2]);
    int num_reps = std::stoi(argv[3]);
    std::string algorithm {argv[4]};
    Reconfig::Algo algo {Util::get_algo_enum(algorithm)};

    // Now read file and convert to JSON object
    Util::JsonWrapper json_file(file_path);

    // Read in initial config, target config from JSON object and problem regions
    int Nt_x = json_file.read_problem_Nt_x();
    int Nt_y = json_file.read_problem_Nt_y();
    const int num_target = json_file.read_problem_num_target();


    std::vector<int32_t>target_config = get_target_config("center compact", Nt_x * Nt_y, num_target);
 
    // check required number of atoms and generate initial configuration
    int required_num = Util::count_num_atoms(target_config);
	double loss_params_alpha = json_file.read_problem_alpha();
	double loss_params_nu = json_file.read_problem_nu();
	double lifetime = json_file.read_problem_lifetime();

    int successes = 0;
    bool failure = false;
    // START LOOP
    for (int trial = 0; trial < num_trials; ++trial) {
        // generate current config
        std::vector<int32_t> trial_config(Nt_x * Nt_y, 0);
        for(auto &it: trial_config) {
            it = (0.52 >= (((double)rand()) / RAND_MAX))? 1: 0;
        }

        // Create solver object from the config
        float extra = 0.0;

        for(int rep = 0; rep < num_reps; ++rep) {
            Reconfig::Solver solver(Nt_x, Nt_y, nullptr);
            std::vector<int32_t> rep_config(trial_config);
            bool not_enough_atoms = false;
            TrapArray trap_array = TrapArray(Nt_y, Nt_x, rep_config, loss_params_alpha , loss_params_nu , lifetime);  
            while (!Util::target_met(rep_config, target_config)) {
                // check if we meet the required target config
                // if we do not, then check if we have enough atoms to continue the process
                if (Util::count_num_atoms(rep_config) < required_num) {
                    not_enough_atoms = true;
                    break;
                }
                
                std::vector<Reconfig::Move> moves_list;
                int ret;
                // start the solver
                solver.start_solver(algo, rep_config, target_config, 0,0 ,0 );
                // generate the moves
                moves_list = solver.gen_moves_list(algo, 0, 0, 0);
                // perform the moves
                if (trap_array.performMoves(moves_list) != 0) {
                    std::cout << "Something went wrong when producing moves" << std::endl;
                    failure = true;
                    break;
                }

                // perform loss
                // trap_array.printTrapArray();
                trap_array.performLoss();
                trap_array.getArray(rep_config);

            }

            if (not_enough_atoms) {
                continue;
            }
            if (failure) {
                break;
            }

            ++successes;
        }
        if (failure) {
            break;
        }
    }
    std::cout << (double)((double)successes/(double)(num_reps * num_trials)) << std::endl;
    return 0;
}

