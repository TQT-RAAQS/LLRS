#ifndef ACQUISITION_STORED_H
#define ACQUISITION_STORED_H

#include "acquisition.h"
#include "llrs-lib/PreProc.h"

namespace Acquisition {
    
    class ImageAcquisitionStored : public ImageAcquisition{
      
      public:
        void setup(uint32_t roi_width, uint32_t roi_height,
                       int acquisition_timeout, uint32_t roi_xoffset = 0,
                       uint32_t roi_yoffset = 0, uint32_t vbin = 1,
                       uint32_t hbin = 1) override {}
        std::vector<uint16_t> acquire_single_image() override;
    };
}
#endif