# Frame Grabber Card

This directory contains the Frame Grabber Card's Cuda driver.

# Build & Usage 
- The driver can be built using the Meson build system. To use this driver in your project, add this module as a subproject with the following specifications:
  1. The project should provide `PROJECT_BASE_DIR` C Preprocessor Macro, it should contain a string of the absolute path of the base directory of the project.
  2. In the `configs/fgc` directory of the project, there should be the `andor_ixonultra888.pcf` (or equivalent) pcf config file provided, an example of this file is provided [here](./andor_ixonultra888.pcf).
