#include "acquisition.h"


void Acquisition::ImageAcquisition::setup(uint32_t roi_width, uint32_t roi_height,
                       int acquisition_timeout, uint32_t roi_xoffset,
                       uint32_t roi_yoffset, uint32_t vbin,
                       uint32_t hbin) {
                        fgc.setup(roi_width, roi_height, acquisition_timeout, roi_xoffset, roi_yoffset, vbin, hbin);
                       }

std::vector<uint16_t> Acquisition::ImageAcquisition::acquire_single_image() {
    std::vector <uint16_t> current_image;
    START_TIMER("I")
    current_image = fgc.acquire_single_image();
    END_TIMER("I")

    if (current_image.empty()) {
        throw std::runtime_error("Image Acquisition Failed.");
    }
    return current_image;
}