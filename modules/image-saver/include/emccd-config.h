#include "globals-config.h"

class EmccdConfig : protected GlobalsConfig {
    
    int roi_x, roi_y, roi_w, roi_h, hbin, vbin;

  public:
    EmccdConfig(ShotFile shotfile)
        : GlobalsConfig(
              shotfile,
              {{"emccd_roi_x", &roi_x, LabscriptType::VALUE},
               {"emccd_roi_y", &roi_y, LabscriptType::VALUE},
               {"emccd_roi_w", &roi_w, LabscriptType::VALUE},
               {"emccd_roi_h", &roi_h, LabscriptType::VALUE},
               {"emccd_hbin", &hbin, LabscriptType::VALUE},
               {"emccd_vbin", &vbin, LabscriptType::VALUE}}) {}

    int get_roi_x() const { return roi_x; }
    int get_roi_y() const { return roi_y; }
    int get_roi_w() const { return roi_w; }
    int get_roi_h() const { return roi_h; }
    int get_hbin() const { return hbin; }
    int get_vbin() const { return vbin; }
};