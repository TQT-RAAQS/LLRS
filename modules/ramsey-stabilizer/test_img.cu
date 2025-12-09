#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include "ImageProcessor.h"
#include <fstream>
#include <fftw3.h>
#include <omp.h>
#include <cmath>
#include <fftw3.h>
#include <vector>
#include <complex>
#include <iostream>

int main() {
    // Reading the image addresses

    boost::filesystem::path exp_directory = \
        LabscriptAddressUtils::get_experiment_directory_from_time("2025-12-01", "12_46_16");
    auto directory_address = exp_directory / boost::filesystem::path("raw_data");

    std::vector<std::string> img_first_addresses, img_second_addreesses;
    for (auto const& shot : boost::filesystem::directory_iterator(directory_address)) {
        if (boost::filesystem::is_directory(shot)) {
            for (auto const& img : boost::filesystem::directory_iterator(shot)) {
                if (boost::filesystem::is_regular_file(img) && img.path().extension() == ".png") {
                    auto& filename = img.path().filename().string();
                    auto identifier = filename.substr(filename.length() - 6);
                    if (identifier == "-0.png") {
                        img_first_addresses.emplace_back(img.path().string());
                    } else if (identifier == "-1.png") {
                        img_second_addreesses.emplace_back(img.path().string());
                    }
                }
            }
        }
    }

    // Image analysis

    Processing::ImageProcessor ip;
    ip.reload();

    const size_t trap_count = ip.get_trap_count();
    std::vector<double_t> fls0(trap_count), fls1(trap_count);
    std::vector<uint8_t> oc0(trap_count), oc1(trap_count);
    std::vector<double> signal(trap_count);

    size_t shot_index = 5;

    auto img0_mat = cv::imread(img_first_addresses.at(shot_index), cv::IMREAD_UNCHANGED);
    auto img1_mat = cv::imread(img_second_addreesses.at(shot_index), cv::IMREAD_UNCHANGED);

    const auto img_width = img0_mat.cols;

    std::vector<uint16_t> img0(img0_mat.total());
    std::vector<uint16_t> img1(img1_mat.total());
    std::memcpy(img0.data(), img0_mat.data, img0_mat.total() * sizeof(uint16_t));
    std::memcpy(img1.data(), img1_mat.data, img1_mat.total() * sizeof(uint16_t));
    
    ip.process(img_width, 0, img0, fls0, oc0);
    ip.process(img_width, 1, img1, fls1, oc1);

    for (size_t i = 0; i < trap_count; ++i) {
        signal[i] = (oc0[i] ? (oc1[i] ? 1 : -1) : 0);
    }

    // Reading orders
    size_t Ny, Nx;
    std::vector<int64_t> orders;

    std::ifstream fin(TRAPS_ORDERS_TRANSLATION_FILE, std::ios::binary);

    fin.read(reinterpret_cast<char*>(&Ny), sizeof(int64_t));
    fin.read(reinterpret_cast<char*>(&Nx), sizeof(int64_t));
    
    orders.resize(Nx*Ny);
    fin.read(reinterpret_cast<char*>(orders.data()), Nx*Ny*sizeof(int64_t));
    
    fin.close();

    // ordering the signal
    std::vector<double> signal_ordered(trap_count);
    for (size_t i = 0; i < trap_count; ++i) {
        signal_ordered[i] = signal[orders[i]];
    }
    
    // fast fourier transform
    std::vector<std::complex<double>> signal_fft(Ny * (Nx/2 + 1));
    auto plan = fftw_plan_dft_r2c_2d(
        Ny,
        Nx,
        signal_ordered.data(),
        reinterpret_cast<fftw_complex*>(signal_fft.data()),
        FFTW_ESTIMATE
    );
    fftw_execute(plan);

    // Peak index
    size_t peak_index = -1;
    double peak = -1;

    for (size_t i = 1; i < signal_fft.size(); ++i) {
        double mag = std::norm(signal_fft[i]);
        if (mag > peak) {
            peak = mag;
            peak_index = i;
        }
    }


    const double dx = 10.6e-6;
    const double dy = 5.3e-6;

    double fx = 1.0 / Nx / dx;
    double fy = 1.0 / Ny / dy;

    const size_t peak_index_x = peak_index % (Nx/2 + 1);
    const size_t peak_index_y = peak_index / (Nx/2 + 1);

    std::cout << peak_index_x*fx << " " << peak_index_y*fy << std::endl;
    std::cout << std::atan2(signal_fft[peak_index].imag(),  signal_fft[peak_index].real()) << std::endl;
    
    fftw_destroy_plan(plan);
    fftw_cleanup();

    return 0;
}