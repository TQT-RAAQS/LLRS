#include "ImageProcessor-replace.h"

std::vector<int32_t> Processing::ImageProcessorReplace::apply_threshold(
    std::vector<double> filtered_vec, double threshold) {

    START_TIMER("II-Threshold");
    std::vector<int32_t> atom_configuration(filtered_vec.size(), 0);
    for (int trap_idx = 0; trap_idx < filtered_vec.size(); trap_idx++) {
        if (filtered_vec[trap_idx] >
            threshold) { // check if the trap contains an atom by comparing
                         // agaisnt the threshhold
            atom_configuration[trap_idx] = 1;
        }
    }
    END_TIMER("II-Threshold");

    return configs.at(cycleCount++);
}