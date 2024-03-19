# Config & Files

The following is the current paths used in LLRS project to config and read data:
 
## AWG

#### `awg.yml` Config File

Reads from the `config/awg` directory, the path is hardwired in the include file corresponding to AWG in the `common-include` directory. This config file is used when `AWG::configure` is called, and the following are the data retrevied from this file:
	1. `driver_path`
	2. `sample_rate`
	3. `external_clock_freq`
	4. `freq_resolution`
	5. `channels`
	6. `amp`
	7. `amp_lim`
	8. `awg_sample_rate`
	9. `wfm_mask`
	10. `waveform_duration`
	11. `waveforms_per_segment`
	12. `null_segment_length`
	13. `idle_segment_length`

## FGC

#### `activesilicon_1xcld.pcf` Config File

Reads from the `config/fgc` directory, the path is hardwired in the implementation file, (`activesilicon-1xcld.cpp`) and the file is accessed when the constructor for the `FGC` class is called. The data that is read from the file are all _FGC_-specific.

## LLRS-lib

### Image-Processing

#### `psf` Data File

Reads form the `resources/psf` directory, the path directory for the path is hardwired in the `PreProc.h` include file corresponding to the `LLRS-lib` in the `common-include` directory, the specific file name is read from the config file supplied to the library when calling `setup()`. The data that is read from file are for the Point Spread Function.

#### `pgm` Image File

Writes the image to the _epoch_-time labelled `pgm` file in the `resources/image` directory, the path is hardwired in the `PreProc.h` include file corresponding to the `LLRS-lib` in the `common-include` directory. The file is written when the `write_to_pgm()` method is called.


### Setup 
#### `coef_x` and `coef_y` Data Files

Reads from the `resources/coef/{primary, secondary}` directories (`coef_x` from `primary` and `coef_y` from `secondary`), the paths for the directories are hardwired in the `PreProc.h` include file corresponding to `LLRS-lib` in the `common-include` directory, and the specific file names are read from the config file supplied to the library when calling `setup()`. This data file is used when `read_fparams` is called (while creating/reading a _Waveform Repo_), and the alpha, nu, phi parameters are retrevied from this file:

#### `waveform-repo` Cache File

Reads from `resources/wfm` directory, the path for the directory is hardwired in the `PreProc.h` include file corresponding to `LLRS-lib` in the `common-include` directory, and the specific file name is determined based on the repo's specifications. When `create_wf_repo()` is called, the method checks if such cached file already exists and reads from it if so. Otherwise, a new cached file is created for current and future use. 

### Library

#### `config.{yml,json}` Config File

Reads from the `configs/llrs-lib` directory, the path for the directory is hardwired in the `PreProc.h` include file corresponding to the `LLRS-lib` in the `common-include` directory, and the specific name is provided to `LLRS` when calling the `setup()` or `small_setup()` methods. The data from the config files is used throughout the _LLRS_ procedure, especially during `setup()`/`small_setup`. The config file can be in either `YAML` or `JSON` and must contain the following parameters:

```YAML
problem_definition:
  experiment_params: 
    roi_width: # Int 
    roi_height: # Int 
    detection_threshold: # Float
    psf:    # psf.bin,     see generate_psf
    coef_x: # default.csv, see generate_coef
    coef_y: # default.csv, see generate_coef
  problem_params:
    Nt_x: # Int 
    Nt_y: # Int
    target_config_label: # {centre_compact, custom}
    target_config: # Array, this param is read only when custom is used. 
    num_target: # Int, this param is read only when centre_compact is used
    array_geometry_type: rectangular lattice
    algorithm: # {liblin_exact_1d.so, liblin_exact_1d_cpu_v2.so, libredrec_v2.so, libredrec_cpu_v3.so, libaro_cpu.so, libredrec_gpu_v3.so, libbird_cpu.so}
```


#### `id.json` Solution Files

Reads from the `resources/operational_solutions` directory, the path for the directory is hardwired in the `PreProc.h` include file corresponding to the `LLRS-lib` in the `common-include` directory, and the specific name _can be_ provided to `LLRS` when calling the `setup()` or `small_setup()` methods. The data from the config files is used __only when the `PRE_SOLVED` macro is on___, replacing the data that is usually fetched from pictures taken (in the `execute()` method). This solution file is in `JSON` and is produced by the `runtime-benchmarking`'s Jupyter Notebook.

#### `id.json` Timing Data File

Writes to the `resources/runtime-benchmarks` directory, the path for the directory is hardwired in the `PreProc.h` include file corresponding to the `LLRS-lib` in the `common-include` directory and the specific _can be_ provided to `LLRS` when calling the `setup()` or `small_setup()` methods. It is produced during the `execute()` method when the `LOGGING_RUNTIME` macro is defined. The file contains timers corresponding to the `LLRS`'s runtime formatted in a `JSON` file.

#### `main-log.txt` Logging File

Writes to the `resources/runtime/logs` directory, the path for the directory is hardwired in the `PreProc.h` include file corresponding to the `LLRS-lib` in the `common-include` directory and the file name is always `main-log.txt`. It is produced during the execution of `LLRS` when `LOGGING_VERBOSE` is defined and contains the logging information. 


## Operational Benchmarking


`TO BE UPDATED`



