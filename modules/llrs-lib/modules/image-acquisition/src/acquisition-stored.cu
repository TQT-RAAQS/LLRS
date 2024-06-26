#include "acquisition-stored.h"

std::vector<uint16_t> Acquisition::ImageAcquisitionStored::acquire_single_image() {
    std::vector <uint16_t> current_image;
    START_TIMER("I")
    current_image = fgc.acquire_single_image();
    END_TIMER("I")


    char image_file[256];
    std::string image_file_path =
        (std::string("") + PROJECT_BASE_DIR +
                "/resources/images/fake-image.png");
    strcpy(image_file, image_file_path.c_str());
    current_image = fgc.acquire_stored_image(image_file);
    return current_image;
}