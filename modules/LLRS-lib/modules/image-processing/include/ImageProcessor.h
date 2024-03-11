/*
 * Author: Laurent Zheng, Wendy Lu
 * Winter 2023
 */
 
#ifndef _IMAGE_PROCESSING_H_
#define _IMAGE_PROCESSING_H_

#include "LLRS-lib/Settings.h"
#include "LLRS-lib/PreProc.h"
#include <omp.h>
#include <fstream>
#include <cstdint>
#include <stdlib.h>
#include <ios>
#include <tuple>
#include <chrono>

namespace Processing {

    using PSF_PAIR = std::pair<size_t, double>;

    // Apply a threshold to the filtered vector to determine which traps contain atoms.
    std::vector<int32_t> apply_threshold(std::vector<double> filtered_vec, double threshold);

    // Write image data to a PGM file in binary format, with the filename being the current epoch time in nanoseconds
    void write_to_pgm(const std::vector<uint16_t>& image, int width, int height);

    class ImageProcessor {
    public:
        // Constructor of ImageProcessor class
        ImageProcessor(std::string psf_file, size_t num_trap);
        ImageProcessor() = default;

        // Applies a filter to the image using the stored PSF and returns an array indicating the presence or 
        // absence of atoms in the corresponding trap.
        std::vector<double> apply_filter(std::vector<uint16_t> * p_input_img);

    private:
        std::vector<std::vector<PSF_PAIR>> _psf;
    };
}

#endif