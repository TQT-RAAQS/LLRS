#include "fourier-analyzer.h"

FourierAnalyzer::FourierAnalyzer(size_t Nx_padded, size_t Ny_padded) : 
    Nx_padded(Nx_padded), Ny_padded(Ny_padded) {
    this->reload_orders();
}

FourierAnalyzer::~FourierAnalyzer() {
    if (fft_plan) {
        fftw_destroy_plan(fft_plan);
    }
}

void FourierAnalyzer::execute_fft_plan(const std::vector<uint8_t>& oc0, const std::vector<uint8_t>& oc1) {
    // Calculate the ternary signal
    double sum = 0.0;
    size_t total = this->Nx * this->Ny;    

    #pragma omp simd reduction(+:sum)
    for (size_t idx = 0; idx < total; ++idx) {
        const auto oind = this->orders[idx];
        auto corrected_idx = (idx % this->Nx) + (idx / this->Nx) * this->Nxm;

        int s = oc0[oind] * (static_cast<int>(oc1[oind]) * 2 - 1);

        this->signal[corrected_idx] = static_cast<double>(s);
        sum += static_cast<double>(s);
    }
    
    // Subtract the mean
    double signal_mean = sum / static_cast<double>(total);
    #pragma omp simd
    for (size_t idx = 0; idx < total; ++idx) {
        auto corrected_idx = (idx % this->Nx) + (idx / this->Nx) * this->Nxm;
        this->signal[corrected_idx] -= signal_mean;
    }

    // Take the 2D fourier transform
    fftw_execute(this->fft_plan);
}

std::tuple<int, double, double> FourierAnalyzer::find_fft_peak() {
    size_t peak_index = -1;
    double peak = -1;
    const size_t N = this->signal_fft.size();
    
    for (size_t i = 0; i < N; ++i) {
        double mag = std::norm(this->signal_fft[i]);
        if (mag > peak) {
            peak = mag;
            peak_index = i;
        }
    }

    size_t ix = peak_index % (this->Nxm / 2 + 1);
    size_t iy = peak_index / (this->Nxm / 2 + 1);

    double fx = static_cast<double>(ix)/(this->Nxm);
    double fy = static_cast<double>(iy)/(this->Nym);

    return {peak_index, fx, fy};
}

std::complex<double> FourierAnalyzer::perform_dfft(double fx, double fy) {
    double val_re = 0.0, val_im = 0.0;

    #pragma omp simd reduction(+:val_re, val_im)
    for (size_t idx = 0; idx < this->Nx * this->Ny; ++idx) {
        auto corrected_idx = (idx % this->Nx) + (idx / this->Nx) * this->Nxm;
        double phase = -2.0 * M_PI * (fx * (idx % this->Nx) + fy * (idx / this->Nx));
        val_re += this->signal[corrected_idx] * std::cos(phase);
        val_im += this->signal[corrected_idx] * std::sin(phase);
    }

    return {val_re, val_im};
}

double FourierAnalyzer::cost_function(const std::vector<double>& x, std::vector<double>& grad, void* f_data) {
    (void)grad;
    auto* self = static_cast<FourierAnalyzer*>(f_data);
    return -std::norm(self->perform_dfft(x[0], x[1]));
}

double FourierAnalyzer::extract_phase(const std::vector<uint8_t>& oc0, const std::vector<uint8_t>& oc1) {
    // Perform FFT
    this->execute_fft_plan(oc0, oc1);
    
    // Extract argmax of FFT
    auto peak_info = this->find_fft_peak();
    auto fx_argmax = std::get<1>(peak_info);
    auto fy_argmax = std::get<2>(peak_info);
    
    // Perform Nelder-Mead optimization to refine the peak location
    nlopt::opt opt(nlopt::LN_NELDERMEAD, 2); // 2 variables, no gradient
    double maximum_norm;
    std::vector<double> optimal_f = {fx_argmax, fy_argmax};
    opt.set_min_objective(FourierAnalyzer::cost_function, this);
    opt.optimize(optimal_f, maximum_norm);
    
    // Extract the phase
    double phi = std::arg(this->perform_dfft(optimal_f[0], optimal_f[1]));

    // Modify the phase to center the origin on the middle of the trap array
    phi += 2.0 * M_PI * (optimal_f[0] * this->x0 + optimal_f[1] * this->y0);

    return FourierAnalyzer::wrap_phase(phi);
}

void FourierAnalyzer::reload_orders(const bool flag_translate_psf) {
    if (flag_translate_psf) {
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
    this->x0 = (double)(Nx-1) / 2.0;
    this->y0 = (double)(Ny-1) / 2.0;
}

double FourierAnalyzer::wrap_phase(double phi) {
    // Wrap phase to the range [-pi, pi]
    auto p = std::fmod(phi + M_PI, 2.0 * M_PI);
    return (p > 0 ? p - M_PI : p + M_PI);
}