#include "ramsey-stabilizer.h"

double RamseyStabilizer::wrap_phase(double phi) {
    phi = std::fmod(phi + M_PI, 2.0 * M_PI);
    if (phi < 0)
        phi += 2.0 * M_PI;
    return phi - M_PI;
}

RamseyStabilizer::RamseyStabilizer(const std::string config) {
    this->configs = YAML::LoadFile(RAMSEY_STABILIZER(config));
    this->setup_fft_params();
    this->setup_translator();
    this->setup_memory_handler();
    this->read_orders();
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

void RamseyStabilizer::setup_fft_params() {
    this->Nx_padded = this->configs["spatial_zero_padding_x"].as<size_t>();
    this->Ny_padded = this->configs["spatial_zero_padding_y"].as<size_t>();

    this->dx = this->configs["dx"].as<double>();
    this->dy = this->configs["dx"].as<double>();
}

void RamseyStabilizer::setup_translator() {
    this->flag_configs_translator = this->configs["flag_configs_translater"].as<bool>();
}

void RamseyStabilizer::read_orders() {
    // Re-translate the orders if necessary
    if (this->flag_configs_translator) {
        this->configs_translator.translate_psf();
    }

    // Read the translated file
    std::ifstream fin(TRAPS_ORDERS_TRANSLATION_FILE, std::ios::binary);

    fin.read(reinterpret_cast<char*>(&this->Ny), sizeof(int64_t));
    fin.read(reinterpret_cast<char*>(&this->Nx), sizeof(int64_t));
    
    this->orders.resize(this->Nx*this->Ny);
    fin.read(reinterpret_cast<char*>(this->orders.data()), this->Nx*this->Ny*sizeof(int64_t));
    
    fin.close();

    // Resize the signal vectors
    if (this->fft_plan) {
        fftw_destroy_plan(this->fft_plan);
        this->fft_plan = nullptr;
    }

    this->Nxm = max(this->Nx, this->Nx_padded);
    this->Nym = max(this->Ny, this->Ny_padded);

    this->signal.resize(this->Nxm * this->Nym);
    std::fill(signal.begin(), signal.end(), 0.0);
    this->signal_fft.resize(this->Nym * (this->Nxm/2 + 1));

    // Setup fourier transform plan
    this->fft_plan = fftw_plan_dft_r2c_2d(
        Nym,
        Nxm,
        this->signal.data(),
        reinterpret_cast<fftw_complex*>(this->signal_fft.data()),
        FFTW_MEASURE
    );

    // Re-calculate the origin coordinates
    this->x0 = Nx * dx / 2.0;
    this->y0 = Ny * dy / 2.0;
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

    this->phi = this->find_fft_peak_phase();
    
    // Update the parameter
    auto error = RamseyStabilizer::wrap_phase(this->labscript_config->get_ramsey_stabilizer_phi0() - this->phi);
    this->pid_controller->add_value(error);
}

double RamseyStabilizer::find_fft_peak_phase() {
    // Calculate the ternary signal
    double signal_mean = 0.0;
    for (size_t i = 0; i < this->Ny; ++i) {
        for (size_t j = 0; j < this->Nx; ++j) {
            const auto& oind = this->orders[i*Nx + j];
            this->signal[i*Nx + j] = (this->oc0[oind] == 0 ? 0 : (this->oc1[oind] ? 1 : -1));
            signal_mean += signal[i*Nx + j] / (this->Nx * this->Ny);
        }
    }

    // Subtract the mean
    for (size_t i = 0; i < this->Ny; ++i) {
        for (size_t j = 0; j < this->Nx; ++j) {
            signal[i*Nx + j] -= signal_mean;
        }
    }

    // Take the 2D fourier transform
    fftw_execute(this->fft_plan);

    // Find the peak
    size_t peak_index = -1;
    double peak = -1;
    for (size_t i = 0; i < this->signal_fft.size(); ++i) {
        double mag = std::norm(signal_fft[i]);
        if (mag > peak) {
            peak = mag;
            peak_index = i;
        }
    }

    size_t ix = peak_index % this->Nxm;
    size_t iy = peak_index / this->Nxm;

    double fx = static_cast<double>(ix)/(dx*this->Nxm);
    double fy = static_cast<double>(iy)/(dy*this->Nym);

    // Extract the phase
    const auto& peak_val = this->signal_fft[peak_index];
    double phi = std::atan2(peak_val.imag(), peak_val.real());

    // Modify the phase to center the origin on the middle of the trap array
    phi += 2.0 * M_PI * (fx * this->x0 + fy * this->y0);

    return RamseyStabilizer::wrap_phase(phi);
}

void RamseyStabilizer::transition_to_buffered() {
    this->last_shot_address = this->smh->get_shot_address();
    auto experiment_name = LabscriptAddressUtils::get_experiment_folder_name(this->last_shot_address);

    if (this->last_experiment_folder != experiment_name) { // This is a new experiment
        this->last_experiment_folder = experiment_name;
        
        // Read the shot .h5 file
        this->labscript_config = std::make_unique<RamseyStabilizerLabscriptConfig>(this->last_shot_address);

        // Re-read the geometric ordering of the traps
        this->read_orders();

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

    if (this->fft_plan) {
        fftw_destroy_plan(this->fft_plan);
        this->fft_plan = nullptr;
    }
}

RamseyStabilizer::~RamseyStabilizer() {
    this->stop();
}