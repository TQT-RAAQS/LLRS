#ifndef FOURIER_ANALYZER_H_
#define FOURIER_ANALYZER_H_

#include "configs-translator.h"
#include "llrs-lib/PreProc.h"
#include <vector>
#include <complex>
#include <fftw3.h>
#include <cstdint>
#include <fstream>
#include <cmath>
#include <tuple>
#include <nlopt.hpp>

class FourierAnalyzer {

    size_t Nx, Ny;
    size_t Nx_padded, Ny_padded;
    size_t Nxm, Nym;
    double x0, y0;

    std::vector<int64_t> orders;

    ConfigsTranslator& configs_translator = ConfigsTranslator::instance();

    fftw_plan fft_plan = nullptr;
    std::vector<double> signal;
    std::vector<std::complex<double>> signal_fft;

    std::complex<double> perform_dfft(double fx, double fy);
    static double cost_function(const std::vector<double>& x, std::vector<double>& grad, void* f_data);

    void execute_fft_plan(const std::vector<uint8_t>& oc0, const std::vector<uint8_t>& oc1);
    std::tuple<int, double, double> find_fft_peak();

public:
    FourierAnalyzer(size_t Nx_padded, size_t Ny_padded);
    ~FourierAnalyzer();

    void reload_orders(const bool flag_translate_psf = true);
    double extract_phase(const std::vector<uint8_t>& oc0, const std::vector<uint8_t>& oc1);

    static double wrap_phase(double phi);
};

#endif