/*
 * Author: Laurent Zheng, Wendy Lu
 * Winter 2023
 */

#ifndef IMAGE_PROCESSING_H_
#define IMAGE_PROCESSING_H_

#include "Collector.h"
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

// Write image data to a PGM file in binary format, with the filename being the
// current epoch time in nanoseconds
void write_to_pgm(const std::vector<uint16_t> &image, int width, int height);

class ImageProcessor {
  public:
    void setup(std::string psf_file, size_t num_trap);

    // Applies a filter to the image using the stored PSF and returns an array
    // indicating the presence or absence of atoms in the corresponding trap.
    void apply_filter(std::vector<uint16_t> &p_input_img,
                      std::vector<int32_t> &current_config);
    virtual void apply_threshold(std::vector<int32_t> &current_config,
                                 double threshold);
    virtual ~ImageProcessor() {}

  private:
    std::vector<std::vector<PSF_PAIR>> _psf;
};
} // namespace Processing

#endif
