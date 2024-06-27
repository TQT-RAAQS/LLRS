#include "llrs.h"
LLRS::LLRS(std::shared_ptr<AWG> awg, std::shared_ptr<Acquisition::ImageAcquisition> img_acq, std::shared_ptr<Processing::ImageProcessor> img_proc, std::shared_ptr<Reconfig::Solver> solver)
    : log_out(std::ofstream(LOGGING_PATH(std::string("main-log.txt")),
                            std::ios::app)) {
    std::cout << "LLRS: constructor" << std::endl;
    
    awg_sequence = awg? std::make_unique<Stream::Sequence>(awg, wf_table, Synthesis::read_waveform_duration(WFM_CONFIG_PATH("/config.yml"))) : std::make_unique<Stream::Sequence>(wf_table, Synthesis::read_waveform_duration(WFM_CONFIG_PATH("/config.yml")));
    image_acquisition = img_acq? img_acq : std::make_shared<Acquisition::ImageAcquisition>();
    img_proc_obj = img_proc? img_proc :std::make_shared<Processing::ImageProcessor>();
    solver = solver? solver : std::make_shared<Reconfig::Solver>();
}
/**
 * @brief Sets up the LLRS
 * @param input => config filename
 * @param setup_idle_segment => Is LLRS setting up the idle segment?
 * @param llrs_step_off => index of idle step (sequence memory)
 * @param problem_id => metadata filepath
 */

void LLRS::setup(std::string input, bool setup_idle_segment,
                        int llrs_step_off) {
    std::cout << "LLRS: setup" << std::endl;

    /* Direct std log to file as its buffer */
    old_rdbuf = std::clog.rdbuf();
    std::clog.rdbuf(log_out.rdbuf());

    user_input = Util::JsonWrapper(CONFIG_PATH(input));

    /* Configures size of trap array */
    const size_t Nt_x = user_input.read_problem_Nt_x();
    metadata.setNtx(Nt_x);
    const size_t Nt_y = user_input.read_problem_Nt_y();
    metadata.setNty(Nt_y);
    num_trap = Nt_x * Nt_y;

    /* Configures the target array */
    const size_t num_target = user_input.read_problem_num_target();
    std::string target_config_label =
        user_input.read_problem_target_config_label();
    if (target_config_label == "centre_compact") {
        target_config = get_target_config(CENTRE_COMPACT, num_target);
    } else if (target_config_label == "custom") {
        target_config = user_input.read_problem_target_config();
    } else {
        std::cout << "WARNING: NO TARGET READ" << std::endl;
    }

#ifdef LOGGING_VERBOSE
    INFO << "User Input JSON = " << user_input.get_json_data() << std::endl;
#endif

    /* Camera Settings */
    image_acquisition->setup(
        user_input.read_experiment_roi_width(),
        user_input.read_experiment_roi_height(),
        awg_sequence->get_acq_timeout());

    /* Image Processing Settings */
    img_proc_obj->setup(
        PSF_PATH(user_input.read_experiment_psf_path()), Nt_x * Nt_y);

    detection_threshold = user_input.read_experiment_threshold();

    /* Reconfiguration Parameters */

    algo = Util::get_algo_enum(user_input.read_problem_algo());

    solver->setup(Nt_x, Nt_y, awg_sequence->get_wfm_per_segment());

    /* Waveform Synthesis Initialization */
    wf_table = Setup::create_wf_table(
        Nt_x, Nt_y,
        _2d ? awg_sequence->get_sample_rate() / 2
            : awg_sequence->get_sample_rate(),
        awg_sequence->get_waveform_mask(), awg_sequence->get_vpp(),
        user_input.read_experiment_coefx_path(),
        user_input.read_experiment_coefy_path(), true, false);

    awg_sequence->setup(setup_idle_segment, llrs_step_off, _2d, Nt_x, Nt_y);

    // extra control logic
    cycle_num = 0;
    metadata.reset();
}

void LLRS::reset_psf(std::string psf_file) {
    img_proc_obj->setup(
        PSF_PATH(psf_file), metadata.getNtx() * metadata.getNty());
}

void LLRS::reset_waveform_table() {
    wf_table = Setup::create_wf_table(
        metadata.getNtx(), metadata.getNty(),
        _2d ? awg_sequence->get_sample_rate() / 2
            : awg_sequence->get_sample_rate(),
        awg_sequence->get_waveform_mask(), awg_sequence->get_vpp(),
        user_input.read_experiment_coefx_path(),
        user_input.read_experiment_coefy_path(), true, true);
}


void LLRS::reset_problem(std::string algorithm, int num_target) {
    target_config = get_target_config(CENTRE_COMPACT, num_target);
    algo = Util::get_algo_enum(algorithm);
}

/**
 * @brief Creates desired target atom array, specified in the config
 * file
 * @param target => label of target config
 * @param num_target => number of atoms in target array
 */

std::vector<int32_t> LLRS::get_target_config(Target target,
                                                    int num_target) {
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

/**
 * @brief creates a center compact target array
 * @param num_target => number of atoms in target array
 */

void LLRS::create_center_target(std::vector<int32_t> &target_config,
                                       int num_target) {

    int start_index = (num_trap / 2) - (num_target / 2);

    for (int offset = 0; offset < num_target; ++offset) {
        target_config[start_index + offset] = 1;
    }
}

/**
 * @brief resets the LLRS and metadata between shots
 */
void LLRS::reset(bool reset_segments) {
    metadata.reset();
    awg_sequence->reset(reset_segments);
}

/**
 * @brief executes the LLRS loop
 */
int LLRS::execute() {
    std::cout << "LLRS: execute" << std::endl;
    for (cycle_num = 0; true; ++cycle_num) {

        auto reset_result = std::async(
            std::launch::async, &LLRS::reset, this, false);
#ifdef LOGGING_VERBOSE
        INFO << "Starting cycle " << cycle_num << std::endl;
#endif
        awg_sequence->emccd_trigger();
        std::vector<uint16_t> current_image =
            image_acquisition->acquire_single_image();

#ifdef LOGGING_VERBOSE
        INFO << "Acquired Image of size " << current_image.size()
                << std::endl;
#endif

        reset_result.wait();

#ifdef STORE_IMAGES
        std::thread t_store_image(
            &Processing::write_to_pgm, current_image,
            user_input.read_experiment_roi_width(),
            user_input.read_experiment_roi_height());
        t_store_image.detach();
#endif

        /* Step 1, apply gaussian psf kernel onto each atom position */
        std::vector<double> filtered_output =
            img_proc_obj->apply_filter(&current_image);

        /* Step 2, apply a threshold to the filtered output to get the
            * atom configuration */
        std::vector<int32_t> current_config =
            img_proc_obj->apply_threshold(filtered_output,
                                        detection_threshold);

#ifdef LOGGING_VERBOSE
        INFO << "Filtered output: " << vec_to_str(filtered_output)
                << std::endl;
        INFO << "Current configuration: " << vec_to_str(current_config)
                << std::endl;
#endif
        metadata.addAtomConfigs(current_config);

        if (Util::target_met(current_config, target_config)) {
            INFO << "Success: Met Target Configuration -> exit()"
                    << std::endl;
            metadata.setTargetMet();
            awg_sequence->clock_trigger();
            return 0;
        }

        if (cycle_num >= ATTEMPT_LIMIT) {
#ifdef LOGGING_VERBOSE
            INFO << "Exceeded limit in attempting to solve a trial."
                    << std::endl;
#endif /* Attempt to solve the same problem too many time, exit the loop */
            break;
        }

        START_TIMER("III-Total");
        /* Run the solver with a specified algorithm */
        int solve_status =
            solver->start_solver(algo, current_config, target_config);

        if (solve_status != LLRS_OK) {
#ifdef LOGGING_VERBOSE
            INFO << "Below minimum required number of atoms -> exit()"
                    << std::endl;
#endif /* Target num atom > Current num atoms, exit the loop */
            break;
        }

        /* Translate algorithm output to waveform table key components
            */
        std::vector<Reconfig::Move> moves_list =
            solver->gen_moves_list(algo);
        END_TIMER("III-Total");
        metadata.addMovesPerCycle(moves_list);

#ifdef LOGGING_VERBOSE
        INFO << "Ran Algorithm: " << user_input.read_problem_algo()
                << std::endl;
        INFO << "Source:" << vec_to_str(solver->get_src_atoms())
                << std::endl;
        INFO << "Destination:" << vec_to_str(solver->get_dst_atoms())
                << std::endl;
        INFO << "Batch Indices:" << vec_to_str(solver->get_batch_ptrs())
                << std::endl;
#endif

        GET_EXTERNAL_TIME("IV-Translate", 0);
        awg_sequence->load_and_stream(moves_list);

        solver.reset();
        metadata.incrementNumCycles();
    }

#ifdef LOGGING_RUNTIME
    metadata.addRuntimeData(Util::Collector::get_instance()->get_runtime_data());
    Util::Collector::get_instance()->clear_timers();
#endif

    awg_sequence->clock_trigger();
    return 1;
}

void LLRS::clean() {
    std::clog.rdbuf(old_rdbuf);
    log_out.close();
}

