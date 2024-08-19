# Arbitrary Waveform Generator

This directory contains the Arbitrary Waveform Generator's Cuda driver.

# Build & Usage 
- The driver can be built using the Meson build system. To use this driver in your project, add this module as a subproject with the following specifications:
  1. The project should provide `PROJECT_BASE_DIR` C Preprocessor Macro, it should contain a string of the absolute path of the base directory of the project.
  2. In the `configs/awg` directory of the project, there should be the `config.yml` (or equivalent) config file provided, an example of this file is provided [here](./config.yml).
  3. This module uses `yaml-cpp` as a dependency, this should be provided using Wrap Dependency System as a subproject in the main project.

