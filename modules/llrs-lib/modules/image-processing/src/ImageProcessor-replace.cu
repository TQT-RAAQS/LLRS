#include "ImageProcessor-replace.h"

void Processing::ImageProcessorReplace::apply_threshold(
    std::vector<int32_t> &filtered_vec, double threshold) {

    START_TIMER("II-Threshold");
    for (int trap_idx = 0; trap_idx < filtered_vec.size(); trap_idx++) {
        if (filtered_vec[trap_idx] >
            threshold) { // check if the trap contains an atom by comparing
                         // agaisnt the threshhold
            filtered_vec[trap_idx] = 1;
        } else {
            filtered_vec[trap_idx] = 0;
        }
    }
    END_TIMER("II-Threshold");
    filtered_vec = configs.at(cycleCount++);
}