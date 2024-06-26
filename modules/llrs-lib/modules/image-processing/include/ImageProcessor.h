/*
 * Author: Laurent Zheng, Wendy Lu
 * Winter 2023
 */

#ifndef IMAGE_PROCESSING_H_
#define IMAGE_PROCESSING_H_

#include "llrs-lib/PreProc.h"
#include "llrs-lib/Settings.h"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <ios>
#include <omp.h>
#include <stdlib.h>
#include <tuple>

namespace Processing {

using PSF_PAIR = std::pair<size_t, double>;

// Apply a threshold to the filtered vector to determine which traps contain
// atoms.
std::vector<int32_t> apply_threshold(std::vector<double> filtered_vec,
                                     double threshold);

// Write image data to a PGM file in binary format, with the filename being the
// current epoch time in nanoseconds
void write_to_pgm(const std::vector<uint16_t> &image, int width, int height);

class ImageProcessor {
  public:
    void setup(std::string psf_file, size_t num_trap);

    // Applies a filter to the image using the stored PSF and returns an array
    // indicating the presence or absence of atoms in the corresponding trap.
    std::vector<double> apply_filter(std::vector<uint16_t> *p_input_img);

  private:
    std::vector<std::vector<PSF_PAIR>> _psf;
};
} // namespace Processing

#endif
