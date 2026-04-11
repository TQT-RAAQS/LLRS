#include "ramsey-stabilizer.h"

using namespace MicrowaveHandler;

RamseyStabilizer::RamseyStabilizer(const std::string config) {
    this->configs = YAML::LoadFile(RAMSEY_STABILIZER(config));
    this->setup_awg_handler();
    this->setup_fourier_analyzer();
    this->setup_memory_handler();
    this->setup_saver();
    this->setup_qdac_client();
}

void RamseyStabilizer::setup_qdac_client() {
    auto qdac_config_name = this->configs["qdac_client"]["config"].as<std::string>();
    this->flag_qdac_controller_active = false;
    this->qdac_client = std::make_unique<QdacClient>(qdac_config_name);

    this->gamma = this->configs["magnetometry"]["gamma"].as<double>();
    this->nu_freespace = this->configs["magnetometry"]["nu_fs"].as<double>();
}

void RamseyStabilizer::reset_pid() {
    // General controller parameters
    this->target_phi = this->labscript_config->get_ramsey_stabilizer_phi0();
    this->error = 0;
    this->phi = 0;
    this->pid_count = this->configs["pid_count"].as<size_t>();

    // Clearing past controllers
    this->pid_controllers.clear();
    this->reset_waveform_data();

    // Controller specific initialization
    auto pid_configs = this->configs["pid_config"];
    auto controller_type = static_cast<ControllerType>(this->labscript_config->get_controller_type());

    if (controller_type == ControllerType::PID_PHASE_CONTROLLER) {
        INFO << "Initializing PID phase controller with " << this->pid_count << " loops.\n";

        pid_configs["k_p"] = this->labscript_config->get_ramsey_stabilizer_k_p();
        pid_configs["k_i"] = this->labscript_config->get_ramsey_stabilizer_k_i();
        pid_configs["k_d"] = this->labscript_config->get_ramsey_stabilizer_k_d();
        pid_configs["k_p_width"] = this->labscript_config->get_ramsey_stabilizer_k_p_width();
        pid_configs["max_change"] = this->labscript_config->get_ramsey_stabilizer_max_change();

        for (size_t i = 0; i < this->pid_count; ++i) {
            this->pid_controllers.emplace_back(std::make_unique<PIDLoopPhaseController>(pid_configs));
        }

    } else if (controller_type == ControllerType::LINEAR_CONTROLLER) {
        INFO << "Initializing linear controller with " << this->pid_count << " loops.\n";

        for (size_t i = 0; i < this->pid_count; ++i) {
            this->pid_controllers.emplace_back(std::make_unique<LinearController>(pid_configs));
        }
    } else {
        throw std::runtime_error("Unsupported controller type " + std::to_string(controller_type) + ". Change the controller type in the settings for the Ramsey Stabilizer module.");
    }
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

    this->fourier_analyzer = std::make_unique<FourierAnalyzer>(Nx_padded, Ny_padded);
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
        auto flag_info_log = true;
        while (!this->thread_worker_killed.load()) {
            if (flag_info_log) {
                INFO << "Waiting for new image or shot start signal from shared memory handler...\n";
            }
            auto ret = this->smh->wait_for_update(smh_timeout_s);
            
            if (ret == -1) { // Either an error occurred, or the wait timed out.
                if (errno == ETIMEDOUT) {
                    if (this->thread_worker_killed.load()) {
                        break; // Exit if a stop request was issued.
                    }
                    flag_info_log = false;
                    continue; // Go back to waiting.
                } else {
                    throw std::system_error(errno, std::generic_category(), "Semaphore for the worker failed");
                }
            }
            flag_info_log = true;

            auto image_count = this->smh->get_image_count();
            INFO << "Image count: " << image_count << ", Images processed: " << (int)images_processed << ".\n";

            if (image_count == 0 && images_processed == SHOT_NOT_BEGUN_YET) { // The shot has begun
                INFO << "Transitioning to buffered mode.\n";
                this->transition_to_buffered();
                this->smh->signal_done();
                images_processed = 0;
            } else if (image_count > 0 && image_count > images_processed) { // A new image is available
                INFO << "Processing new image. Image index: " << (int)images_processed << ".\n";
                if (this->flag_active && this->labscript_config->get_ramsey_stabilizer_pid_enabled()) {
                    this->process_image(images_processed);
                }
                ++images_processed;
            } else if (image_count == images_processed) { // The shot is over 
                INFO << "Shot over; processed all images, image count " << image_count << ".\n";
                this->smh->signal_done();
                if (this->flag_active) {
                    this->awg_handler->stop();
                    this->saver->add_to_queue(ShotInformation{
                        this->last_shot_address,
                        this->error,
                        this->waveform_params.at(this->active_pid_index)["nu0"],
                        this->waveform_params.at(this->active_pid_index)["nu0_streamed"],
                        this->moving_average.at(this->active_pid_index),
                        this->awg_handler->get_streaming_time()
                    });
                }
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
    
    // Phase extraction using Fourier analysis
    this->phi = this->fourier_analyzer->extract_phase(this->oc0, this->oc1);
    this->phi *= (2.0 * this->gradient_x_parallel - 1.0); // Adjust for gradient direction along x
    
    // Calculate the true resonance frequency
    auto nu_resonance = this->update_nu0_prime(); // Update the true resonance frequency
    INFO << "Checking if we should sent to qdac client.\n";
    if (this->flag_qdac_controller_active) { // If the QDAC client is active, send the estimated magnetic field to the QDAC server.
        auto b_field = (nu_resonance - this->nu_freespace) / this->gamma;
        this->qdac_client->send_b_field(this->active_pid_index, b_field);
        INFO << "Estimated resonance frequency: " << nu_resonance << " Hz, Estimated magnetic field: " << b_field << " T.\n";
    }
    
    // Update the control parameter
    this->error = FourierAnalyzer::wrap_phase(this->phi - this->target_phi);
    
    auto correction = this->pid_controllers.at(this->active_pid_index)->compute_correction(this->error);
    this->waveform_params.at(this->active_pid_index)["nu0"] += correction;
    INFO << "Extracted phase: " << this->phi << ", Error: " << error << ", Correction: " << correction << ".\n";

    // Track mode to prevent mode hops if enabled
    if (this->flag_track_mode) {
        this->track_mode();
    }
}

double RamseyStabilizer::update_nu0_prime() {
    auto& nu0 = this->waveform_params.at(this->active_pid_index)["nu0_streamed"];
    auto nu_resonance = nu0 - this->phi / (2.0 * M_PI * this->interrogation_tau);
    this->waveform_params.at(this->active_pid_index)["nu0_prime"] = nu_resonance;
    return nu_resonance;
}

void RamseyStabilizer::track_mode() {
    if (this->interrogation_tau <= 0.0 || this->nu_buffer_size == 0) return;
    INFO << "Tracking mode enabled. Current nu0: " << this->waveform_params.at(this->active_pid_index)["nu0"] 
         << " Hz, Moving average: " << this->moving_average.at(this->active_pid_index) << " Hz.\n";

    const auto i = this->active_pid_index;

    auto& buf = this->nu_buffer[i];
    auto& ma  = this->moving_average[i];
    auto& params = this->waveform_params[i];

    auto nu0 = params["nu0"];
    auto max_mode_distance = 1.0 / this->interrogation_tau * this->track_mode_factor;

    if (buf.size() == this->nu_buffer_size && std::abs(nu0 - ma) > max_mode_distance) {
        INFO << "Mode hop event detected. Setting the waveform parameter to the moving average value "
             << ma << " Hz, which is outside the allowed distance "
             << max_mode_distance << " Hz from the current value "
             << nu0 << " Hz.\n";

        nu0 = ma;
        params["nu0"] = nu0;
    }

    INFO << "Adding nu0 value " << nu0 << " Hz to the buffer for moving average calculation.\n";
    buf.push_back(nu0);

    if (buf.size() == this->nu_buffer_size + 1) { // Buffer overflew by one element
        INFO << "Buffer exceeded maximum size of " << this->nu_buffer_size << "; it is " << buf.size() << ". Removing oldest value and updating moving average.\n";
        ma += buf.back() / this->nu_buffer_size - buf.front() / this->nu_buffer_size;
        buf.pop_front();
    } else if (buf.size() <= this->nu_buffer_size) { // Buffer is still not full
        INFO << "Buffer size is " << buf.size() << ". Updating moving average with new value.\n";
        ma = ma * (buf.size() - 1) / buf.size() + buf.back() / buf.size();
    } else {
        throw std::runtime_error("Unexpected case! Buffer size is " + std::to_string(buf.size()) + " but it should never exceed " + std::to_string(this->nu_buffer_size + 1) + ".");
    }
}

void RamseyStabilizer::transition_to_buffered() {
    this->last_shot_address = this->smh->get_shot_address();
    auto experiment_name = LabscriptAddressUtils::get_experiment_folder_name(this->last_shot_address);
    
    // Read the shot .h5 file
    this->labscript_config = std::make_unique<RamseyStabilizerLabscriptConfig>(this->last_shot_address);

    this->flag_active = this->labscript_config->get_ramsey_stabilizer_active_flag();
    this->nu_buffer_size = this->labscript_config->get_ramsey_stabilizer_nu_buffer_size();
    this->flag_track_mode = this->labscript_config->get_ramsey_stabilizer_track_mode();
    this->interrogation_tau = this->labscript_config->get_ramsey_stabilizer_tau();
    this->track_mode_factor = this->labscript_config->get_ramsey_stabilizer_track_mode_factor();

    this->flag_qdac_controller_active = this->labscript_config->get_qdac_controller_active();
    this->error = 0;
    
    if (!this->flag_active) {
        return;
    }

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
    this->moving_average.clear();
    this->nu_buffer.clear();
    
    for (size_t i = 0; i < this->pid_count; ++i) {
        auto is_empty = this->waveform_params.at(i).find("nu0") == this->waveform_params.at(i).end();
        auto initialization_needed = is_empty || this->labscript_config->get_ramsey_stabilizer_clear_memory_flag();

        if (initialization_needed) {
            // Variables that should only be read from labscript if initialiation is needed.
            this->waveform_params.at(i)["nu0"] = this->labscript_config->get_ramsey_stabilizer_nu0();
            this->waveform_params.at(i)["nu0_streamed"] = this->labscript_config->get_ramsey_stabilizer_nu0();
            this->waveform_params.at(i)["nu0_prime"] = this->labscript_config->get_ramsey_stabilizer_nu0();
        }

        // Variables that should always be read from labscript
        this->waveform_params.at(i)["alpha"] = this->labscript_config->get_ramsey_stabilizer_alpha();

        // Track mode logic
        this->nu_buffer.emplace_back();
        this->moving_average.push_back(0.0);
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

void RamseyStabilizer::register_streamed_parameters(std::unordered_map<std::string, double>& vars) {
    vars["nu0_streamed"] = vars["nu0"];
}

std::string RamseyStabilizer::substitute_variables_in_signal(std::string s, std::unordered_map<std::string, double>& vars) {
    for (const auto& kv : vars) {
        std::string key = "$" + kv.first + "$";
        std::string val = std::to_string(kv.second);

        size_t pos = 0;
        while ((pos = s.find(key, pos)) != std::string::npos) {
            s.replace(pos, key.size(), val);
            pos += val.size();
        }
    }
    RamseyStabilizer::register_streamed_parameters(vars); // Register the streamed parameters after substitution.
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
    this->thread_worker_killed.store(true);
    if (this->thread_worker != nullptr && this->thread_worker->joinable()) {
        this->thread_worker->join();
    }
    if (this->awg_handler->is_connected()) {
        this->awg_handler->close_connection();
    }
    this->saver->stop();
}

RamseyStabilizer::~RamseyStabilizer() {
    this->stop();
}