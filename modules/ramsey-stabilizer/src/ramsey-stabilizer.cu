#include "ramsey-stabilizer.h"

RamseyStabilizer::RamseyStabilizer(const std::string config) {
    this->configs = YAML::LoadFile(RAMSEY_STABILIZER(config));
    this->setup_fft();
    this->setup_memory_handler();
    this->setup_saver();
    this->reset_pid();
}

void RamseyStabilizer::reset_pid() {
    auto pid_configs = this->configs["pid_config"];
    pid_configs["param_min"] = this->labscript_config->get_ramsey_stabilizer_delta_min();
    pid_configs["param_max"] = this->labscript_config->get_ramsey_stabilizer_delta_max();
    pid_configs["k_p"] = this->labscript_config->get_ramsey_stabilizer_k_p();
    pid_configs["k_i"] = this->labscript_config->get_ramsey_stabilizer_k_i();
    pid_configs["k_d"] = this->labscript_config->get_ramsey_stabilizer_k_d();
    pid_configs["param_initial"] = 0;

    this->pid_controller = std::make_unique<PIDLoopController>(pid_configs);
}

void RamseyStabilizer::setup_saver() {
    this->saver = std::make_unique<RamseyStabilizerMetadataSaver>(this->configs["saver_config"]);
}

void RamseyStabilizer::setup_fft() {
    auto Nx_padded = this->configs["spatial_zero_padding_x"].as<size_t>();
    auto Ny_padded = this->configs["spatial_zero_padding_y"].as<size_t>();

    auto dx = this->configs["dx"].as<double>();
    auto dy = this->configs["dx"].as<double>();

    this->phase_extractor = std::make_unique<PhaseExtractor>(dx, dy, Nx_padded, Ny_padded);
    this->flag_configs_translator = this->configs["flag_configs_translater"].as<bool>();
}

void RamseyStabilizer::setup_memory_handler() {
    auto handler_config_name = this->configs["memory_handler_config"].as<std::string>();
    this->smh = std::make_unique<SharedMemoryHandler>(handler_config_name);
    this->smh->open_connection();
}

void RamseyStabilizer::worker_function() {
    auto smh_timeout_s = this->configs["memory_handler_wait_s"].as<uint8_t>();
    int8_t images_processed = SHOT_NOT_BEGUN_YET;

    try {
        while (!this->thread_worker_killed.load()) {
            auto ret = this->smh->wait_for_update(smh_timeout_s);
            
            if (ret == -1) { // Either an error occured, or the wait timed out.
                if (errno == ETIMEDOUT) {
                    if (this->thread_worker_killed.load()) break; // Exit if a stop request was issued.
                    continue; // Go back to waiting.
                } else {
                    throw std::system_error(errno, std::generic_category(), "Semaphore for the worker failed");
                }
            }

            auto image_count = this->smh->get_image_count();

            if (image_count == 0 && images_processed == SHOT_NOT_BEGUN_YET) { // The shot has begun
                this->transition_to_buffered();
                this->smh->signal_done();
                images_processed = 0;
            } else if (image_count > 0 && image_count > images_processed) { // A new image is available
                this->process_image(images_processed);
                ++images_processed;
            } else if (image_count == images_processed) { // The shot is over
                this->saver->add_to_queue(this->last_shot_address, this->phi, this->delta, this->pid_controller->get_control_param());
                this->smh->signal_done();
                images_processed = SHOT_NOT_BEGUN_YET;
            } else {
                throw std::runtime_error("Unexpected case in the memory manager of the ramsey stabilizer shared memory handler. This is most likely a bug. Current image count: " + \
                    std::to_string(image_count) + ", processed image count: " + std::to_string(images_processed) + ".");
            }
        }
    } catch (const std::exception& e) {
        INFO << "Unexpected failure. Signalling done and exiting.\n";
        for (size_t i = 0; i < 10; ++i) {
            this->smh->signal_done();
        }
        throw;
    }
}

void RamseyStabilizer::process_image(int8_t image_index) {
    if (image_index > 1) {
        INFO << "Shot has more than two images. This is unexpected; this module will ignore all images after the second one.\n";
        return;
    }

    // Read the atom occupancy states
    auto occ = this->smh->get_trap_occupancy(image_index);

    if (image_index == 0) { // If this is the first image
        this->oc0 = std::move(occ);
        return;
    }

    // If this is the second image
    this->oc1 = std::move(occ);

    this->phi = this->phase_extractor->extract_phase(this->oc0, this->oc1);
    
    // Update the parameter
    auto error = PhaseExtractor::wrap_phase(this->labscript_config->get_ramsey_stabilizer_phi0() - this->phi);
    this->pid_controller->add_value(error);
}

void RamseyStabilizer::transition_to_buffered() {
    this->last_shot_address = this->smh->get_shot_address();
    auto experiment_name = LabscriptAddressUtils::get_experiment_folder_name(this->last_shot_address);

    if (this->last_experiment_folder != experiment_name) { // This is a new experiment
        this->last_experiment_folder = experiment_name;
        
        // Read the shot .h5 file
        this->labscript_config = std::make_unique<RamseyStabilizerLabscriptConfig>(this->last_shot_address);

        // Re-read the geometric ordering of the traps
        this->phase_extractor->read_orders(this->flag_configs_translator);

        // Reset pid params
        this->reset_pid();
    }

    this->delta = this->pid_controller->get_control_param();
}

void RamseyStabilizer::start() {
    this->thread_worker_killed.store(false);
    this->thread_worker = std::make_unique<std::thread>(&RamseyStabilizer::worker_function, this);
    this->saver->start();
}

void RamseyStabilizer::stop() {
    this->thread_worker_killed.store(true);
    if (this->thread_worker != nullptr && this->thread_worker->joinable()) {
        this->thread_worker->join();
    }
    this->saver->stop();
}

RamseyStabilizer::~RamseyStabilizer() {
    this->stop();
}