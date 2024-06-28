#ifndef ACQUISITION_STORED_H
#define ACQUISITION_STORED_H

#include "acquisition.h"
#include "llrs-lib/PreProc.h"

namespace Acquisition {
    
    class ImageAcquisitionStored : public ImageAcquisition{
      
      public:
        std::vector<uint16_t> acquire_single_image() override;
    };
}
#endif