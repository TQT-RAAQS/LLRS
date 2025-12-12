#ifndef PHASE_EXTRACTOR_H_
#define PHASE_EXTRACTOR_H_

#include "configs-translator.h"
#include "llrs-lib/PreProc.h"
#include <vector>
#include <complex>
#include <fftw3.h>
#include <cstdint>
#include <fstream>
#include <cmath>

class PhaseExtractor {

    size_t Nx, Ny;
    size_t Nx_padded, Ny_padded;
    size_t Nxm, Nym;
    double dx, dy;
    double x0, y0;

    std::vector<int64_t> orders;

    ConfigsTranslator& configs_translator = ConfigsTranslator::instance();

    fftw_plan fft_plan = nullptr;
    std::vector<double> signal;
    std::vector<std::complex<double>> signal_fft;

public:
    PhaseExtractor(double dx, double dy, size_t Nx_padded, size_t Ny_padded);
    ~PhaseExtractor();

    void reload_orders(const bool flag_translate_psf = true);
    double extract_phase(const std::vector<uint8_t>& oc0, const std::vector<uint8_t>& oc1);

    static double wrap_phase(double phi);
};

#endif