#include "linear-controller.h"

LinearController::LinearController(YAML::Node configs){
    this->force_reload = configs["reload_needed"].as<bool>();
    this->max_buffer_size = configs["max_buffer_size"].as<size_t>();
    this->reset();
}

void LinearController::reset(){
    // Clearing the queue
    this->error_buffer.clear();
    this->correction_buffer.clear();

    // Reading the file again to reset the coefficients as well, in case they were updated.
    auto address = LINEAR_CONTROLLER_TRANSLATION_FILE;
    auto file_exists = fs::exists(address);

    if (this->force_reload || !file_exists) {
        INFO << "Reloading linear controller coefficients from translator.";
        this->translator.translate_linear_controller_configs();
    } else {
        INFO << "Linear controller coefficients file already exists. Skipping translation.";
    }

    // Reading the file

    std::ifstream fin(address, std::ios_base::in | std::ios_base::binary);

    if (!fin.is_open()) {
        throw std::invalid_argument("Linear controller coefficients file not found");
    }

    fin.read(reinterpret_cast<char*>(&this->M), sizeof(this->M)); // Number of coefficients.
    fin.read(reinterpret_cast<char*>(&this->lp_alpha), sizeof(this->lp_alpha)); // Low-pass filter alpha.

    this->error_coefficients.resize(this->M);
    this->correction_coefficients.resize(this->M - 1);

    for (size_t i = 0; i < this->M; ++i) {
        fin.read(reinterpret_cast<char*>(&this->error_coefficients[i]), sizeof(double));
    }
    
    for (size_t i = 0; i < this->M - 1; ++i) {
        fin.read(reinterpret_cast<char*>(&this->correction_coefficients[i]), sizeof(double));
    }

    fin.close();

    if (this->max_buffer_size < this->M) {
        throw std::invalid_argument("Max buffer size must be at least M.");
    }
}

double LinearController::compute_correction(double error) {
    // Apply low-pass filter to the error
    auto lp_error = this->error_buffer.size() > 0 ? 
        this->lp_alpha * error + (1 - this->lp_alpha) * this->error_buffer.front() : 
        error;
    
    // Add error to the error buffer
    this->error_buffer.push_front(lp_error);
    if (this->error_buffer.size() > this->max_buffer_size) {
        this->error_buffer.pop_back();
    }

    // Calculate correction term
    auto correction = this->calculate_correction_term();
    this->correction_buffer.push_front(correction);
    if (this->correction_buffer.size() > this->max_buffer_size) {
        this->correction_buffer.pop_back();
    }
    
    return correction;
}

double LinearController::calculate_correction_term() {
    if (this->error_buffer.size() < this->M) {
        // Not enough data to calculate correction
        return 0.0;
    }

    double correction = 0.0;
    for (size_t i = 0; i < this->M - 1; ++i) {
        correction += this->error_coefficients.at(i) * this->error_buffer.at(i);
        correction += this->correction_coefficients.at(i) * this->correction_buffer.at(i);
    }
    correction += this->error_coefficients.at(M-1) * this->error_buffer.at(M-1);

    return correction;
}