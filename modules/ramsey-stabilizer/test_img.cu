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
#include "ramsey-stabilizer-labscript-config.h"

int main() {
    std::string address = "/home/tqtraaqs/Z/Experiments/Rydberg/2025-12-12/18_17_29-test_emccd_camera/labscript_shot_outputs/test_emccd_camera_2025-12-12_0009_0.h5";
    RamseyStabilizerLabscriptConfig config(address);

    std::cout << config.get_ramsey_stabilizer_first_gate_phase() << std::endl;

    return 0;
}