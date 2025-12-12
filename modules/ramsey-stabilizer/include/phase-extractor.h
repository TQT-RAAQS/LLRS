#ifndef PHASE_EXTRACTOR_H_
#define PHASE_EXTRACTOR_H_

#include <vector>
#include <complex>
#include <fftw3.h>
#include <cstdint>

class PhaseExtractor {
private:
    size_t Nx, Ny;
    size_t Nx_padded, Ny_padded;
    size_t Nxm, Nym;
    double dx, dy;
    double x0, y0;

    fftw_plan fft_plan = nullptr;
    std::vector<double> signal;
    std::vector<std::complex<double>> signal_fft;

public:
    PhaseExtractor(size_t Nx, size_t Ny, double dx, double dy, double x0, double y0);
    ~PhaseExtractor();

    void setup_fft_params();
    void process_image(const std::vector<uint8_t>& image, int8_t image_index);
    double find_fft_peak_phase();

    static double wrap_phase(double phi);
};

#endif