#include "ramsey-stabilizer.h"

using namespace MicrowaveHandler;

RamseyStabilizer::RamseyStabilizer(const std::string config) {
    this->configs = YAML::LoadFile(RAMSEY_STABILIZER(config));
    this->setup_awg_handler();
    this->setup_fourier_analyzer();
    this->setup_memory_handler();
    this->setup_saver();
}

void RamseyStabilizer::reset_pid() {
    auto pid_configs = this->configs["pid_config"];
    pid_configs["k_p"] = this->labscript_config->get_ramsey_stabilizer_k_p();
    pid_configs["k_i"] = this->labscript_config->get_ramsey_stabilizer_k_i();
    pid_configs["k_d"] = this->labscript_config->get_ramsey_stabilizer_k_d();
    pid_configs["max_change"] = this->labscript_config->get_ramsey_stabilizer_max_change();

    this->target_phi = this->labscript_config->get_ramsey_stabilizer_phi0();
    this->error = 0;
    this->phi = 0;
    this->pid_count = this->configs["pid_config"]["pid_count"].as<size_t>();

    this->pid_controllers.clear();
    for (size_t i = 0; i < this->pid_count; ++i) {
        this->pid_controllers.emplace_back(std::make_unique<PIDLoopPhaseController>(pid_configs));
    }

    this->reset_waveform_data();
}

void RamseyStabilizer::setup_awg_handler() {
    const auto& c = this->configs["awg_handler"];

    auto awg_config_name = c["config"].as<std::string>();
    this->awg_handler = std::make_unique<MicrowaveAwgHandler>(awg_config_name);
}

void RamseyStabilizer::setup_saver() {
    this->saver = std::make_unique<RamseyStabilizerMetadataSaver>(this->configs["saver_config"]);
}

void RamseyStabilizer::setup_fourier_analyzer() {
    const auto& c = this->configs["fourier_analyzer"];

    auto Nx_padded = c["spatial_zero_padding_x"].as<size_t>();
    auto Ny_padded = c["spatial_zero_padding_y"].as<size_t>();

    auto dx = c["dx"].as<double>();
    auto dy = c["dy"].as<double>();

    this->fourier_analyzer = std::make_unique<FourierAnalyzer>(dx, dy, Nx_padded, Ny_padded);
    this->flag_configs_translator = this->configs["flag_configs_translator"].as<bool>();
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
            
            if (ret == -1) { // Either an error occurred, or the wait timed out.
                if (errno == ETIMEDOUT) {
                    if (this->thread_worker_killed.load()) {
                        break; // Exit if a stop request was issued.
                    }
                    continue; // Go back to waiting.
                } else {
                    throw std::system_error(errno, std::generic_category(), "Semaphore for the worker failed");
                }
            }

            auto image_count = this->smh->get_image_count();
            INFO << "Image count: " << image_count << ", Images processed: " << (int)images_processed << ".\n";

            if (image_count == 0 && images_processed == SHOT_NOT_BEGUN_YET) { // The shot has begun
                INFO << "Transitioning to buffered mode.\n";
                this->transition_to_buffered();
                this->smh->signal_done();
                images_processed = 0;
            } else if (image_count > 0 && image_count > images_processed) { // A new image is available
                INFO << "Processing new image. Image index: " << (int)images_processed << ".\n";
                if (this->labscript_config->get_ramsey_stabilizer_pid_enabled()) {
                    this->process_image(images_processed);
                }
                ++images_processed;
            } else if (image_count == images_processed) { // The shot is over
                INFO << "Shot is over. Adding metadata to queue.\n";
                this->smh->signal_done();
                this->awg_handler->stop();
                this->saver->add_to_queue(this->last_shot_address, this->error, this->waveform_params.at(this->active_pid_index)["nu0"]);
                this->labscript_config.reset();
                images_processed = SHOT_NOT_BEGUN_YET;
            } else {
                INFO << "Unexpected case in the memory manager. Current image count: " << image_count
                     << ", Processed image count: " << (int)images_processed << ".\n";
                throw std::runtime_error("Unexpected case in the memory manager of the ramsey stabilizer shared memory handler. This is most likely a bug. Current image count: " + \
                    std::to_string(image_count) + ", processed image count: " + std::to_string(images_processed) + ".");
            }
        }
    } catch (const std::exception& e) {
        INFO << "Unexpected failure: " << e.what() << ". Signalling done and exiting.\n";
        for (size_t i = 0; i < 10; ++i) {
            this->smh->signal_done();
        }
        throw;
    }

    INFO << "Worker function exiting.\n";
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
    
    this->phi = this->fourier_analyzer->extract_phase(this->oc0, this->oc1);
    this->phi *= (2.0 * this->gradient_x_parallel - 1.0); // Adjust for gradient direction along x
    
    // Update the parameter
    this->error = FourierAnalyzer::wrap_phase(this->phi - this->target_phi);
    auto correction = this->pid_controllers.at(this->active_pid_index)->compute_correction(this->error);
    INFO << "Extracted phase: " << this->phi << ", Error: " << error << ", Correction: " << correction << ".\n";

    this->waveform_params.at(this->active_pid_index)["nu0"] += correction;
}

void RamseyStabilizer::transition_to_buffered() {
    this->last_shot_address = this->smh->get_shot_address();
    auto experiment_name = LabscriptAddressUtils::get_experiment_folder_name(this->last_shot_address);
    
    // Read the shot .h5 file
    this->labscript_config = std::make_unique<RamseyStabilizerLabscriptConfig>(this->last_shot_address);
    
    if (this->last_experiment_folder != experiment_name) { // This is a new experiment
        this->last_experiment_folder = experiment_name;
        
        // Re-read the geometric ordering of the traps
        this->fourier_analyzer->reload_orders(this->flag_configs_translator);
        
        // Reset PID parameters
        this->reset_pid();
        
        this->gradient_x_parallel = this->labscript_config->get_ramsey_stabilizer_gradient_x_parallel();
    }
    
    this->active_pid_index = this->labscript_config->get_ramsey_stabilizer_active_pid_index();
    if (this->active_pid_index >= this->pid_count) {
        throw std::runtime_error("Active PID index " + std::to_string(this->active_pid_index) + " is out of range (PID count: " + std::to_string(this->pid_count) + "). Change the number of PID loops in the settings for the Ramsey Stabilizer module.");
    }
    this->prepare_awg();
}

void RamseyStabilizer::reset_waveform_data() {
    if (this->waveform_params.size() != this->pid_count) {
        this->waveform_params.clear();
        this->waveform_params.resize(this->pid_count);
    }
    
    for (size_t i = 0; i < this->pid_count; ++i) {
        auto is_empty = this->waveform_params.at(i).find("nu0") == this->waveform_params.at(i).end();
        auto initialization_needed = is_empty | this->labscript_config->get_ramsey_stabilizer_clear_memory_flag();

        if (initialization_needed) {
            // Variables that should only be read from labscript if initialiation is needed.
            this->waveform_params.at(i)["nu0"] = this->labscript_config->get_ramsey_stabilizer_nu0();
        }

        // Variables that should always be read from labscript
        this->waveform_params.at(i)["alpha"] = this->labscript_config->get_ramsey_stabilizer_alpha();
    }
}

void RamseyStabilizer::prepare_awg() {
    std::vector<MicrowaveHandler::Waveform> waveforms;
    
    // String analysis
    auto signals = this->labscript_config->get_mw_signals();
    auto substituted_signal = RamseyStabilizer::substitute_variables_in_signal(signals, this->waveform_params.at(this->active_pid_index));
    auto signal_tokens = RamseyStabilizer::split_signal(substituted_signal, ';');
    for (const auto& s : signal_tokens) {
        waveforms.push_back(MicrowaveHandler::waveform_from_string(s));
    }

    // Upload the waveforms
    this->awg_handler->upload_waveforms(waveforms);

    // Start the AWG
    this->awg_handler->start();
}

std::string RamseyStabilizer::substitute_variables_in_signal(std::string s, const std::unordered_map<std::string, double>& vars) {
    for (const auto& kv : vars) {
        std::string key = "$" + kv.first + "$";
        std::string val = std::to_string(kv.second);

        size_t pos = 0;
        while ((pos = s.find(key, pos)) != std::string::npos) {
            s.replace(pos, key.size(), val);
            pos += val.size();
        }
    }
    return s;
}

std::vector<std::string> RamseyStabilizer::split_signal(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, delim)) {
        if (!item.empty())
            tokens.push_back(item);
    }
    return tokens;
}

void RamseyStabilizer::start() {
    this->awg_handler->open_connection();
    this->thread_worker_killed.store(false);
    this->thread_worker = std::make_unique<std::thread>(&RamseyStabilizer::worker_function, this);
    this->saver->start();
}

void RamseyStabilizer::stop() {
    if (this->awg_handler->is_connected()) {
        this->awg_handler->close_connection();
    }
    this->thread_worker_killed.store(true);
    if (this->thread_worker != nullptr && this->thread_worker->joinable()) {
        this->thread_worker->join();
    }
    this->saver->stop();
}

RamseyStabilizer::~RamseyStabilizer() {
    this->stop();
}