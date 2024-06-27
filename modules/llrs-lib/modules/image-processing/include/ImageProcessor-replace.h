#ifndef IMAGE_PROCESSING_REPLACE_H_
#define IMAGE_PROCESSING_REPLACE_H_

#include "ImageProcessor.h"

namespace Processing {

class ImageProcessorReplace : public ImageProcessor {
    std::vector<std::vector<int32_t>> configs;
    int cycleCount = 0;
  public:
    void set_configs(std::vector<std::vector<int32_t>> replacement) {configs = replacement; cycleCount = 0;}
    std::vector<int32_t> apply_threshold(std::vector<double> filtered_vec,
                                     double threshold) override;
};
} // namespace Processing



#endif
