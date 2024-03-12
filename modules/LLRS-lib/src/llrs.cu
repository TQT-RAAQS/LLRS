#include "llrs.h"

template<typename AWG_T> LLRS<AWG_T>::LLRS(): log_out(std::ofstream(LOGGING_PATH(std::string("main_log.txt")), std::ios::app)), 
p_collector {Util::Collector::get_instance()},
awg_sequence{std::make_unique<Stream::Sequence<AWG_T>>(p_collector, wf_table)} {
    std::cout << "LLRS: constructor" << std::endl;
}

template<typename AWG_T> LLRS<AWG_T>::LLRS(std::shared_ptr<AWG_T> &awg): log_out(std::ofstream(LOGGING_PATH(std::string("main_log.txt")), std::ios::app)), 
p_collector {Util::Collector::get_instance()},
awg_sequence{std::make_unique<Stream::Sequence<AWG_T>>(awg, p_collector, wf_table)} {
    std::cout << "LLRS: constructor" << std::endl;
}


template<typename AWG_T>void LLRS<AWG_T>::small_setup( std::string json_input ){
    std::cout << "LLRS: setup" << std::endl;

    /* Direct std log to file as its buffer */
    old_rdbuf = std::clog.rdbuf();
    std::clog.rdbuf(log_out.rdbuf());
   
    user_input = Util::JsonWrapper((json_input));

    const size_t num_target = user_input.read_problem_num_target();
    std::string target_config_label = user_input.read_problem_target_config_label();

    if (target_config_label == "centre_compact"){
        target_config = get_target_config(CENTRE_COMPACT, num_target);
    } else if (target_config_label == "custom") {
        target_config = user_input.read_problem_target_config(); 
    } else {
        std::cout << "WARNING: NO TARGET READ" << std::endl;
    }
    std::cout << "TARGET CONFIG: " << vec_to_str(target_config) << std::endl; 
    /* Image Processing Settings */
    img_proc_obj = Processing::ImageProcessor(
        PSF_PATH(user_input.read_experiment_psf_path()), 
        metadata.getNtx() * metadata.getNty()
    );

    detection_threshold = user_input.read_experiment_threshold();

    /* Reconfiguration Parameters */

    algo = Util::get_algo_enum(user_input.read_problem_algo());

    solver = Reconfig::Solver(
        metadata.getNtx(), metadata.getNty(), p_collector
    );


}

template<typename AWG_T>void LLRS<AWG_T>::setup( std::string input, size_t llrs_seg_off, size_t llrs_step_off, std::string problem_id ){
   std::cout << "LLRS: setup" << std::endl;

    /* Direct std log to file as its buffer */
    old_rdbuf = std::clog.rdbuf();
    std::clog.rdbuf(log_out.rdbuf());
    this->problem_id = problem_id;

	user_input = Util::JsonWrapper(CONFIG_PATH(input));

    const size_t Nt_x = user_input.read_problem_Nt_x();
    metadata.setNtx(Nt_x);
    const size_t Nt_y = user_input.read_problem_Nt_y();
    metadata.setNty(Nt_y);
    num_trap = Nt_x * Nt_y;
    const size_t num_target = user_input.read_problem_num_target();
    std::string target_config_label = user_input.read_problem_target_config_label();

    if (target_config_label == "centre_compact"){
        target_config = get_target_config(CENTRE_COMPACT, num_target);
    } else if (target_config_label == "custom") {
        target_config = user_input.read_problem_target_config(); 
    } else {
        std::cout << "WARNING: NO TARGET READ" << std::endl;
    }


#ifdef LOGGING_VERBOSE
    INFO << "User Input JSON = " << user_input.get_json_data() << std::endl; 
#endif

    /* Camera FGC Settings */
    // DKEA: remove out of llrs eventually
    fgc = std::make_unique<Acquisition::ActiveSilicon1XCLD>( user_input.read_experiment_roi_width(), user_input.read_experiment_roi_height() );
    
    /* Image Processing Settings */
    img_proc_obj = Processing::ImageProcessor(
        PSF_PATH(user_input.read_experiment_psf_path()), 
        Nt_x * Nt_y
    );

    detection_threshold = user_input.read_experiment_threshold();

    /* Reconfiguration Parameters */

    algo = Util::get_algo_enum(user_input.read_problem_algo());

    solver = Reconfig::Solver(
        Nt_x, Nt_y, p_collector
    );

    /* Waveform Synthesis Initialization */
    const double table_sample_rate = _2d? AWG_SAMPLE_RATE/2:AWG_SAMPLE_RATE;

    wf_table = Setup::create_wf_table(
        Nt_x, Nt_y,
        table_sample_rate,
        WAVEFORM_DUR,
        user_input.read_experiment_coefx_path(),
        user_input.read_experiment_coefy_path(),
        false 
    );

    awg_sequence->setup(llrs_seg_off, llrs_step_off, _2d, Nt_x, Nt_y);

    // extra control logic
    trial_num           = 0;
    rep_num             = 0;
    cycle_num           = 0;
    metadata.setNumCycles(0);
    metadata.setMovesPerCycle({});
    metadata.setAtomConfigs({});

}

template<typename AWG_T> std::vector<int32_t> LLRS<AWG_T>::get_target_config(Target target, int num_target ){
    std::vector<int32_t> target_config(num_trap, 0);

    switch (target) {
        case CENTRE_COMPACT:
            create_center_target(target_config, num_target);
            break;
        default:
        std::cerr << "ERROR: Desired configuration not available" << std::endl;
        throw std::invalid_argument("Desired configuration not available");
    }
    return target_config;
}
template<typename AWG_T>void LLRS<AWG_T>::create_center_target( std::vector<int32_t> &target_config, int num_target ){
    
    int start_index = (num_trap / 2) - (num_target / 2);

    for (int offset = 0; offset < num_target; ++offset) {
        target_config[start_index + offset] = 1;
    }
    
}


template<typename AWG_T>void LLRS<AWG_T>::reset(){
    awg_sequence->reset(); 
    trial_num           = 0;
    rep_num             = 0;
    cycle_num           = 0;
    metadata.setNumCycles(0);
    metadata.setMovesPerCycle({});
    metadata.setAtomConfigs({});

}

template<typename AWG_T>void LLRS<AWG_T>::execute(){
    std::cout << "LLRS: execute" << std::endl;
    reset();

#ifdef PRE_SOLVED
    /* Loop through all of the solutions when pre-solved is toggled on */
    Json::Value problem_soln = Util::read_json_file(SOLN_PATH(problem_id));
    for (trial_num = 0; problem_soln.isMember(TRIAL_NAME(trial_num)); trial_num++){
        Json::Value trial_soln = problem_soln[TRIAL_NAME(trial_num)];
        for (rep_num = 0; trial_soln.isMember(REP_NAME(rep_num)); rep_num++){
            Json::Value rep_soln = trial_soln[REP_NAME(rep_num)];
#endif
        
            for(cycle_num = 0; true; cycle_num++){

                awg_sequence->pre_load();


#ifdef LOGGING_VERBOSE
                INFO << "~~~~~~~~~~~~Starting cycle " << cycle_num << " of repetition " 
                << rep_num << " of trial " << trial_num << "~~~~~~~~~~~~" << std::endl;
#endif

#ifdef LOGGING_RUNTIME
                p_collector->start_timer("I", trial_num, rep_num, cycle_num);
#endif

#ifndef PRE_SOLVED
                awg_sequence->emccd_trigger();
                std::vector<uint16_t> current_image = fgc->acquire_single_image();
                // char image_file[256];
                // strcpy(image_file, "/home/tqtraaqs2/Z/Experiments/Rydberg/2023-12-12/12_45_23-atom_imaging/raw_data/atom_imaging_2023-12-12_0003_000/image_2023-12-12_124633775765.png");
                // std::vector<uint16_t> current_image = fgc->acquire_stored_image(image_file);

                //DEBUG: Making fake image
                // std::vector<uint16_t> current_image(user_input.read_experiment_roi_width() *
                //                                    user_input.read_experiment_roi_height(),
                //                                    0);

#ifdef LOGGING_RUNTIME
                p_collector->end_timer("I", trial_num, rep_num, cycle_num);
#endif
                    
                if (current_image.empty()){
                    ERROR << "Image Acquisition Failed." << std::endl;
                    break;
                }

#ifdef LOGGING_VERBOSE
                INFO << "Acquired Image of size " << current_image.size() << std::endl;
#endif

#ifdef STORE_IMAGES
                std::thread t_store_image (
                    &Processing::write_to_pgm,
                    current_image,
                    user_input.read_experiment_roi_width(),
                    user_input.read_experiment_roi_height()
                );
                t_store_image.detach();
#endif

#ifdef LOGGING_RUNTIME
                p_collector->start_timer("II-Deconvolution", trial_num, rep_num, cycle_num);
#endif
                /* Step 1, apply gaussian psf kernel onto each atom position */
                std::vector<double> filtered_output = img_proc_obj.apply_filter(&current_image);
#ifdef LOGGING_RUNTIME
                p_collector->end_timer("II-Deconvolution", trial_num, rep_num, cycle_num);
#endif

#ifdef LOGGING_RUNTIME
                p_collector->start_timer("II-Threshold", trial_num, rep_num, cycle_num);
#endif
                /* Step 2, apply a threshold to the filtered output to get the atom configuration */
                std::vector<int32_t> current_config = Processing::apply_threshold(
                    filtered_output, detection_threshold
                );


#ifdef LOGGING_RUNTIME
                p_collector->end_timer("II-Threshold", trial_num, rep_num, cycle_num);
#endif

# ifdef LOGGING_VERBOSE
                INFO << "Filtered output: " << vec_to_str(filtered_output) << std::endl;
# endif 

#endif
#ifdef PRE_SOLVED
#ifdef LOGGING_RUNTIME
                p_collector->end_timer("I", trial_num, rep_num, cycle_num);
                p_collector->start_timer("II-Deconvolution", trial_num, rep_num, cycle_num);
                p_collector->end_timer("II-Deconvolution", trial_num, rep_num, cycle_num);
                p_collector->start_timer("II-Threshold", trial_num, rep_num, cycle_num);
                p_collector->end_timer("II-Threshold", trial_num, rep_num, cycle_num);
#endif
 
                /* SWAP with pre-solved from processed */
                std::vector<int32_t> current_config = Util::vector_transform(rep_soln[CYCLE_NAME(cycle_num)]);
#endif 

#ifdef LOGGING_VERBOSE
                INFO << "Current configuration: " << vec_to_str(current_config) << std::endl;
#endif
                metadata.addAtomConfigs(current_config);


                if (Util::target_met(current_config, target_config)){
                    INFO << "Success: Met Target Configuration -> exit()" << std::endl;
                    break;
                }

                if (cycle_num >= ATTEMPT_LIMIT){
#ifdef LOGGING_VERBOSE
                    INFO << "Exceeded limit in attempting to solve a trial." << std::endl;
#endif              /* Attempt to solve the same problem too many time, exit the loop */
                    break;
                }

#ifdef LOGGING_RUNTIME
                p_collector->start_timer("III-Total", trial_num, rep_num, cycle_num);
#endif
                /* Run the solver with a specified algorithm */
                int solve_status = solver.start_solver(algo, current_config, target_config, trial_num, rep_num, cycle_num);
               if (solve_status != LLRS_OK){
#ifdef LOGGING_VERBOSE
                    INFO << "Below minimum required number of atoms -> exit()" << std::endl;
#endif              /* Target num atom > Current num atoms, exit the loop */
#ifdef LOGGING_RUNTIME
                p_collector->end_timer("III-Total", trial_num, rep_num, cycle_num);
#endif
                    break;
                }

#ifdef LOGGING_VERBOSE
                INFO << "Ran Algorithm: " << user_input.read_problem_algo() << std::endl;
                INFO << "Source:" << vec_to_str(solver.get_src_atoms()) << std::endl;
                INFO << "Destination:" << vec_to_str(solver.get_dst_atoms()) << std::endl;
                INFO << "Batch Indices:" << vec_to_str(solver.get_batch_ptrs()) << std::endl;
#endif

               /* Translate algorithm output to waveform table key components */

                std::vector<Reconfig::Move> moves_list = solver.gen_moves_list(algo, trial_num, rep_num, cycle_num);

#ifdef LOGGING_RUNTIME
                p_collector->end_timer("III-Total", trial_num, rep_num, cycle_num);
                p_collector->start_timer("IV-Translate", trial_num, rep_num, cycle_num);
#endif
 
                metadata.moves_per_cycle.push_back({});
                metadata.moves_per_cycle.back().insert(metadata.moves_per_cycle.back().end(), moves_list.begin(), moves_list.end());

                awg_sequence->load_and_stream(moves_list, trial_num, rep_num, cycle_num);
               
                /* Makes sure the null sequence loops itself after all steps are taken */
               /* Clean up the solver buffer */
                solver.reset();

            }

#ifdef PRE_SOLVED
        }
    }
#endif 

#ifdef LOGGING_RUNTIME
    /* Log all runtime data as a json based on probelm id */
    std::string output_fname = BENCHMARK_PATH(problem_id);
    Util::write_json_file(p_collector->gen_runtime_json(), output_fname);
#endif

    metadata.num_cycles++;

    return;
}


template<typename AWG_T>int LLRS<AWG_T>::getTrialNum(){
    return trial_num;
}

template<typename AWG_T>int LLRS<AWG_T>::getRepNum(){
    return rep_num;
}


template<typename AWG_T>LLRS<AWG_T>::Metadata::Metadata() {}

// Define Metadata getter functions
template<typename AWG_T>const int LLRS<AWG_T>::Metadata::getNtx() const{
    return Nt_x;
}

template<typename AWG_T>const int LLRS<AWG_T>::Metadata::getNty() const{
    return Nt_y;
}

template<typename AWG_T>const int LLRS<AWG_T>::Metadata::getNumCycles() const {
    return num_cycles;
}

template<typename AWG_T>const std::vector<std::vector<Reconfig::Move>>& LLRS<AWG_T>::Metadata::getMovesPerCycle() const {
    return moves_per_cycle;
}

template<typename AWG_T>const std::vector<std::vector<int32_t>>& LLRS<AWG_T>::Metadata::getAtomConfigs() const {
    return atom_configs;
}

// Define Metadata setter functions
template<typename AWG_T>void LLRS<AWG_T>::Metadata::setNtx(const int new_Ntx) {
    Nt_x = new_Ntx;
}

template<typename AWG_T>void LLRS<AWG_T>::Metadata::setNty(const int new_Nty) {
    Nt_y = new_Nty;
}

template<typename AWG_T>void LLRS<AWG_T>::Metadata::setNumCycles(const int cycles) {
    num_cycles = cycles;
}

template<typename AWG_T>void LLRS<AWG_T>::Metadata::setMovesPerCycle(const std::vector<std::vector<Reconfig::Move>>& moves) {
    moves_per_cycle = moves;
}

template<typename AWG_T>void LLRS<AWG_T>::Metadata::setAtomConfigs(const std::vector<std::vector<int32_t>>& configs) {
    atom_configs = configs;
}

template<typename AWG_T>void LLRS<AWG_T>::setTargetConfig(std::vector<int> new_target_config){
    target_config = new_target_config;
}

template<typename AWG_T>void LLRS<AWG_T>::Metadata::addAtomConfigs(const std::vector<int32_t>& atom_config) {
    std::vector<std::vector<int32_t>> currentAtomConfigs = getAtomConfigs();
    currentAtomConfigs.push_back(atom_config);
    setAtomConfigs(currentAtomConfigs);
}

template<typename AWG_T>void LLRS<AWG_T>::clean(){

    std::clog.rdbuf(old_rdbuf);
    log_out.close();

    // DKEA: free allocated transfer buffers here
    delete p_collector;
    // fgc->destroy_handle();
    // delete fgc;
}

template class LLRS<AWG>;


template<typename AWG_T>void LLRS<AWG_T>::set_psf(std::string json_input){

    old_rdbuf = std::clog.rdbuf();
    std::clog.rdbuf(log_out.rdbuf());
    
    user_input = Util::JsonWrapper((json_input));

    img_proc_obj = Processing::ImageProcessor(
        PSF_PATH(user_input.read_experiment_psf_path()), 
        user_input.read_problem_Nt_x() * user_input.read_problem_Nt_y()
    );
}

template<typename AWG_T>void LLRS<AWG_T>::set_target(std::string json_input){

    old_rdbuf = std::clog.rdbuf();
    std::clog.rdbuf(log_out.rdbuf());
    
    user_input = Util::JsonWrapper((json_input));

    const size_t num_target = user_input.read_problem_num_target();
    std::string target_config_label = user_input.read_problem_target_config_label();

    if (target_config_label == "centre_compact"){
        target_config = get_target_config(CENTRE_COMPACT, num_target);
    } else if (target_config_label == "custom") {
        target_config = user_input.read_problem_target_config(); 
    }


}

