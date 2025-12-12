#include "phase-extractor.h"
#include <cmath>
#include <stdexcept>

PhaseExtractor::PhaseExtractor(size_t Nx, size_t Ny, double dx, double dy, double x0, double y0)
    : Nx(Nx), Ny(Ny), dx(dx), dy(dy), x0(x0), y0(y0) {
    // Initialize FFTW plan or other setup if needed
}

PhaseExtractor::~PhaseExtractor() {
    if (fft_plan) {
        fftw_destroy_plan(fft_plan);
    }
}

void PhaseExtractor::setup_fft_params() {
    // Setup FFT parameters and plan
    // Example:
    signal.resize(Nx * Ny);
    signal_fft.resize(Nx * (Ny / 2 + 1));
    fft_plan = fftw_plan_dft_r2c_2d(Nx, Ny, signal.data(),
                                    reinterpret_cast<fftw_complex*>(signal_fft.data()), FFTW_MEASURE);
}

void PhaseExtractor::process_image(const std::vector<uint8_t>& image, int8_t image_index) {
    // Process the image and populate `signal`
    // Example:
    if (image.size() != Nx * Ny) {
        throw std::runtime_error("Image size does not match expected dimensions.");
    }
    for (size_t i = 0; i < image.size(); ++i) {
        signal[i] = static_cast<double>(image[i]);
    }

    // Execute FFT
    fftw_execute(fft_plan);
}

double PhaseExtractor::find_fft_peak_phase() {
    // Find the peak in the FFT and calculate the phase
    size_t peak_index = 0;
    double max_magnitude = 0.0;

    for (size_t i = 0; i < signal_fft.size(); ++i) {
        double magnitude = std::abs(signal_fft[i]);
        if (magnitude > max_magnitude) {
            max_magnitude = magnitude;
            peak_index = i;
        }
    }

    return std::arg(signal_fft[peak_index]);
}

double PhaseExtractor::wrap_phase(double phi) {
    // Wrap phase to the range [-pi, pi]
    while (phi > M_PI) phi -= 2 * M_PI;
    while (phi < -M_PI) phi += 2 * M_PI;
    return phi;
}