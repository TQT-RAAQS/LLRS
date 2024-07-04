# LLRS
The Low-Latency Reconfiguration System (LLRS) is a closed-loop feedback control system that arranges and reconfigures atoms within an array of laser traps.

## System Outline 

The LLRS contains five submodules:
```mermaid 
graph TD
A[Image Acquisition] -->|image of the trap array| B[Image Processing]  
B -->|binary vector of atom presences in each trap| C[Problem Solving] 
C -->|list of ordered and batched moves to be performed| D[Waveform Synthesis] 
D -->|waveforms representing the atom movements| E[Waveform Streaming] 
E -->| new reconfiguration of atoms in the trap trap array| A 
```
Each of these submodules is defined in [`modules/llrs-lib/modules`](https://github.com/TQT-RAAQS/LLRS/tree/main/modules/llrs-lib/modules).

## LLRS Directory 

```
LLRS
├─ .gitignore
├─ README.md
├─ LICENSE 
├─ resources
├─ configs
│  ├─ llrs 
│  ├─ awg
│  ├─ emccd 
│  ├─ fgc 
│  ├─ waveforms
│  ├─ waveform-power-safety
│  ├─ runtime-benchmarking 
│  └─ operational-benchmarking
├─ tools
│  ├─ setup-emccd.ipynb
│  ├─ generate-fake-psf.py 
│  ├─ psf-translator.py 
│  ├─ algorithm-animation-generator.ipynb
│  └─ benchmarks 
│     ├─ operational-benchmarking.ipynb
│     ├─ runtime-benchmarking.ipynb
│     ├─ image-acquisition-characterization.ipynb 
│     ├─ image-processing-characterization.ipynb 
│     └─ wfm-streaming-characterization.ipynb
└─ modules
    ├─ awg    [External Dependency]
    ├─ fgc    [External Dependency] 
    ├─ llrs-lib
    │  ├─ image-acquisition
    │  ├─ image-processing
    │  ├─ reconfiguration
    │  ├─ solver 
    │  ├─ waveform-synthesis
    │  ├─ waveform-streaming
    │  ├─ setup 
    │  ├─ collector 
    │  ├─ utility 
    │  └─ jsonwrapper 
    ├─ llcs 
    ├─ llrs-exe 
    ├─ operational-benchmarking
    ├─ runtime-benchmarking
    ├─ waveform-loader 
    ├─ hdf5-wrapper 
    ├─ movement-generator 
    └─ zmq-server 
```


## Installation & Setup

## Hardware Requirements

The LLRS requires the PC with the following hardware specifications to operate:

### Processors
- x86 64-bit CPU 
- Nvidia GPU with CUDA support
### Operating System
- Unix-based OS (Developed and tested on Ubuntu 18.04 LTS)  
### Peripherals
- Frame Grabber Card (FGC):
The Frame Grabber Card must be installed on a PCIe slot and connected to the EMCCD camera via a Camera Link cable. Additionally, the camera must be connected via a trigger (MMCX) cable to the AWG trigger port (multipurpose IO port) for image triggering.

- Arbitrary Waveform Generator (AWG):
The Arbitrary Waveform Generator card must be installed on a PCIe slot and its analog output channels be connected to the respective AODs via a SMA cable.

```mermaid
erDiagram
    "CPU-based motherboard" }|..|{ GPU : PCIe
    "CPU-based motherboard" }|..|{ FGC : PCIe
    "CPU-based motherboard" }|..|{ AWG : PCIe
    FGC ||--|{ EMCCD : "Camera Link"
    EMCCD ||--|{ AWG : "MMCX Cable"
    AODs ||--|{ AWG : "SMA Cable"
```

## Software Requirements

Before usage, the following software must be installed and compiled to have a LLRS executable:

### Requirements

- GCC | The C++ compiler that is used to compile LLRS. Pre-installed on most GNU/Linux distro. | [Installation Guide](https://gcc.gnu.org/install/)
 
- NVCC CUDA Compiler | The Cuda compiler that is used to compile the LLRS. Requires a CUDA-enabled Nvidia GPU.| [Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
 
- Nvidia CUDA Toolkit | The CUDA library is used only for the GPU features of the LLRS. | [Installation Guide](https://developer.nvidia.com/cuda-downloads)
 
- Meson Build System | The build system used to compile the project. | [Installation Guide](https://mesonbuild.com/Getting-meson.html)

- Python3 (Version 3.8 or newer) | Only required for benchmarking. | [Installation Guide](https://wiki.python.org/moin/BeginnersGuide/Download)
 
- Jupyter Notebook | Only required for benchmarking. | [Installation Guide](https://jupyter.org/install)
 
### Peripheral Drivers

- FGC:
The LLRS uses ActiveSilicon FireBird Stick FGC and hence requires the installation of the ActiveSilicon SDK. It can be obtained from [ActiveSilicon](https://www.activesilicon.com/products/ActiveSDK-software-development-kit/), and must be then installed on the system. LLRS' build system will look for the SDK in the default installation path at `/usr/local/activesilicon`, but this can be changed in the corresponding [Meson file](modules/fgc/ActiveSDKv01.08.02/meson.build).

- AWG: 
Similar to the FGC, the LLRS uses Spectrum Instrumentation's AWG cards and hence requires the installation of the Spectrum Instrumentation driver and SDKs. They can be obtained from [Spectrum Instrumentation](https://spectrum-instrumentation.com/support/knowledgebase/software/How_to_compile_the_Linux_Kernel_Driver.php), and must be then installed on the system. LLRS' build system will look for the Spectrum library in the default installation paths.

- EMCCD:
Although the LLRS doesn't directly interact with the EMCCD, the EMCCD is required to be set up and waiting for hardware triggers before running the LLRS. The exact drivers and function calls that are necessary for this depends on your camera's make and model. An example for such a procedure using the Andor's Python SDKs for the Andor iXon Ultra 888 can be found in the `tools/setup-emccd.ipynb` notebook.

### Build the LLRS
To compile the LLRS, follow these steps:
- Clone the repo on your local machine and navigate to the LLRS directory.
- In the terminal, execute the command `meson setup bin` in the LLRS directory.
- Navigate into the bin directory using `cd bin`.
- Execute `meson compile` to compile the LLRS.
- The compiled libraries and executables can be found under the `bin/modules` directory.

## Usage
LLRS requires multiple experiment-specific configurations and resources in order to run:

### Resources 
#### PSF 

The LLRS utilizes the images acquired by the frame grabber card to determine the position of atoms and reconfigure them. To process these images the LLRS requires the user to define the position of each trap and the expected fluoresence from atoms to detect the vacant and occupied traps.

LLRS utilizes a set of point-spread functions corresponding to each trap and a universal threshold to determine if the traps are occupied by atoms or are vacant. Put simply, for each trap a set of pixels and their associated weights are specified. The image processing algorithm calculates the weighted average of each of these pixels and compares them against the threshold. Averages above this threshold correspond to an occupied trap, and averages below the threshold correspond to a vacant trap.

Example: Suppose for trap $0$ we define the pixel indices $4098$, $4099$, $4100$ and the associated weights $0.25, 0.5, 0.25$. The universal threshold is $580$. The input image to the image processing module is a $1024 \times 1024$ image. The indices mentioned correspond to the first $p_1$, second $p_2$, and third $p_3$ pixels on the third row of the image (e.g. $4100 = 1024 \times 2 + 2$, making it the third pixel on the third row). Suppose $p_1=580, p_2=600, p_2 = 580$. The weighted average is calculated as: $0.25 p_1 + 0.5 p_2 + 0.25 p_3 = 590$. The result is compared against the universal threshold $580$, and since $590 > 580$, the module concludes that trap $0$ is occupied by an atom.

Here we describe the format in which the threshold and the point spread functions are stored.

configs/llrs/default.yml:
```yml
problem_definition:
    experiment_params:
        detection_threshold: 580
        psf: psf.bin # name of the file under resources/psf/
resources/psf/psf.bin: This is a binary file. A list of numbers are stored on this file in the following order: trap index 1 (type int), pixel index 1 (type int), weight 1 (type double), trap index 2 (type int), pixel index 2 (type int), weight 2 (type double), ... . Note that this is a binary file, meaning that all the numbers are written consecutively with no delimiters.
```

Example:
```
0 4098 0.25 # in order: trap index, pixel index, weight
0 4099 0.50
0 4100 0.25
1 4110 0.25
1 4111 0.5
1 4112 0.2
1 4113 0.05
```
This binary file corresponds to two traps. The first trap (index 0) has three pixels with indices 4098, 4099, and 4100. The corresponding weights are $0.25, 0.5, 0.25$. Likewise, the second trap (index 1) has four pixels with indices 4110, 4111, 4112, and 4113. The corresponding weights are $0.25, 0.5, 0.2, 0.05$.

#### Coefficients 
The LLRS utilizes optical tweezers generated by one or a pair of acousto-optical deflectors (i.e. dynamic traps) to extract, move, and implant atoms in an existing array of optical tweezers (i.e. static traps). To use the LLRS in your system, the user needs to specify the waveforms streamed to the acousto-optical deflectors to generate the traps.

When the dynamic traps are generated and left in an idle state (not moving), the waveform streamed by the acousto-optical deflectors (AODs) are as follows:

$$ f_x(t) = \sum_{x,i} \alpha_i \sin(2\pi \nu_{x,i} t + \phi_{x,i}) $$

$$ f_y(t) = \sum_{y,i} \alpha_i \sin(2\pi \nu_{y,i} t + \phi_{y,i}) $$

$f_x$ and $f_y$ are the waveforms supplied to the $x$ and $y$ AODs. We define the units of $f_x$ and $f_y$ such that they are normalized, i.e. $f(t) = \pm 1$ corresponds to the AWG streaming the maximum voltage configured in awg.yml. Example: If the peak to peak voltage of the AWG is set to $160\ \text{mV}$, $f_x = 0.8$ implies a voltage output of $64\ \text{mV}$ from the AWG.
$\alpha_i$ are the coefficients corresponding to $i$th trap generated by each AOD, in the same units as $f$.
$\nu_i$ are the frequencies corresponding to the $i$th trap generted by each AOD, in units of MHz.
$\phi_i$ is the phase corresponding to the $i$th trap generated by each AOD, defined in the range $[-\pi, \pi]$.
The LLRS requiers the idle waveforms to be fully defined by the user prior to execution. This information is used to generate transition/extraction/implanatation waveforms automatically.

Under the directories resources/coef/primary and resources/coef/secondary (for the first and second AODs respectively) save the confiurations related to the idle waveforms in a .csv file of your choosing in the following format. Afterwards, change the llrs config file under configs/llrs to use the file you defined under resources/coef.

Example:

configs/llrs/default.yml:
```yml
problem_definition:
    experiment_params:
        coef_x: coef_x.yml
        coef_y: coef_y.yml
```
resources/coef/primary/coef_x.csv:
```csv
alpha,nu,phi
<alpha_0>,<nu_0>,<phi_0>
<alpha_1>,<nu_1>,<phi_1>
...
<alpha_N>,<nu_N>,<phi_N>
```
resources/coef/primary/coef_y.csv: 
```csv
alpha,nu,phi 
<alpha_0>,<nu_0>,<phi_0> 
<alpha_1>,<nu_1>,<phi_1> 
... 
<alpha_N>,<nu_N>,<phi_M>
```

### Configurations

#### Waveforms

The waveforms corresponding to the movement along the traps can be configured at `configs/waveforms/config.yml`. A more detailed explanation of the configuration parameters can be found in the Waveform Synthesis' [README file](modules/llrs-lib/modules/waveform-synthesis/README.md).

#### AWG

The AWG configuration can be found at `configs/awg/awg.yml`. This file contains the configuration for the AWG-related parameters, e.g., the nominal peak-to-peak voltage, the sampling rate, and the number of segments.

#### AWG Power Calculator

There is a need for predicting the power streamed by the AWG (and potentially amplified by an electronic amplifier) before supplying it to the target device, e.g. AOMs or AODs. To perform this calculation and to ensure the safety of the device, the following information must be provided. This information can be found by 'calibrating' the AWG.

To do this, follow the following procedure.

1. Keep track of the nominal peak-to-peak voltage configured for the AWG channel you wish to calibrate at `config/awg/awg.yml`.

2. Connect the AWG output to the Spectrum Analyzer. If you wish to use an electronic amplifier in the final setup, you can connect the AWG output to the amplifier and connect the amplifier output to the Spectrum Analyzer instead.

3. Stream a monotonic waveform at some frequency with the maximum amplitude possible. 

4. Fill out the config file at `configs/waveform-power-calculator/config.yml`

Note that this calculator assumes the gain of the AWG/amplifiers for different frequencies is identical.

#### FGC

The FGC requires a PCF file containing the information about the camera connected to the card. This file must be stored under the `configs/fgc` directory.

#### LLRS

All parameters related to the LLRS can be modified in the `configs/llrs/llrs.yml` file. This file contains the configuration for the LLRS-related parameters, e.g., the target config and the algorithm used. It also contains the path to the PSF file and the coefficient files.

**Note**: The LLRS is set up to search the proper subdirectories for necessary support files. All resource files must be stored at the appropriate subdirectory, and only the file name must be provided in the field, not the full file path. 

### Execution

There are several methods for running the LLRS.

#### LLRS Executable
Run the LLRS executable placed at `bin/modules/llrs-exe/llrs-exe`. 
This is the simplest way to run the LLRS. The LLRS-exe will wait for a hardware trigger to the AWG, and then execute the LLRS sequence. The LLRS-exe will then reset and wait for the next hardware trigger to re-execute the LLRS. 

#### LLCS (Low Latency Control System)

This Finite State Machine-based system is designed to be integrated with LabScript Suit via a ZMQ server. Details can be found at the [LLCS README](modules/llcs/README.md).

#### Design your own experiment-specific control system using the LLRS libraries

The LLRS shared libraries (`.so`) are designed to be modular and can be used to design your own control system that suits your experiment's needs. The shared library can be found at `bin/modules/llrs-lib/`, and can be installed on the system using `meson install`.

