#ifndef ACQUISITION_H
#define ACQUISITION_H
#include "Collector.h"
#include "activesilicon-1xcld.hpp"

namespace Acquisition {
    
    class ImageAcquisition {
      protected:
        ActiveSilicon1XCLD fgc;

      public:
        virtual void setup(uint32_t roi_width, uint32_t roi_height,
                       int acquisition_timeout, uint32_t roi_xoffset = 0,
                       uint32_t roi_yoffset = 0, uint32_t vbin = 1,
                       uint32_t hbin = 1);
        virtual std::vector<uint16_t> acquire_single_image();
        virtual ~ImageAcquisition() {};
    };
}
#endif