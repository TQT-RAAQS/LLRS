/*
 * Author: Laurent Zheng, Wendy Lu
 * Winter 2023
 */

#ifndef IMAGE_PROCESSING_H_
#define IMAGE_PROCESSING_H_

#include "Collector.h"
#include "llrs-lib/PreProc.h"
#include "llrs-lib/Settings.h"
#include "configs-translator.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <omp.h>
#include <stdlib.h>
#include <tuple>
#include <thread>

namespace Processing {

using PSF_PAIR = std::pair<size_t, double>;
using PSF_TUPLE = std::tuple<size_t, size_t, double>; // y_index, x_index, weight

// Write image data to a PGM file in binary format, with the filename being the
// current epoch time in nanoseconds
void write_to_pgm(const std::vector<uint16_t> &image, int width, int height);

class ImageProcessor {

  ConfigsTranslator& configs_translator = ConfigsTranslator::instance();

  // DEPRECATED
  std::vector<std::vector<PSF_PAIR>> _psf;

  std::vector<std::vector<PSF_TUPLE>> psfs;
  std::vector<std::vector<double_t>> thresholds;

  void parse_file(std::ifstream &infile);

public:
    void reload(bool flag_translate_psf = true);
    void process(size_t image_width,
                 size_t image_index,
                 const std::vector<uint16_t>& image,
                 std::vector<double_t>& fls_counts,
                 std::vector<uint8_t>& occupancy);
    size_t get_trap_count();

    // DEPRECATED
    void setup(std::string psf_file, size_t num_trap);

    // DEPRECATED
    // Applies a filter to the image using the stored PSF and returns an array
    // indicating the presence or absence of atoms in the corresponding trap.
    void apply_filter(std::vector<uint16_t> &p_input_img,
                      std::vector<int32_t> &current_config);

    // DEPRECATED
    virtual void apply_threshold(std::vector<int32_t> &current_config,
                                 double threshold);

    ImageProcessor();
};
} // namespace Processing

#endif
