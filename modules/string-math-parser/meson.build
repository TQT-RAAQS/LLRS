project(
    'string-math-parser',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
    subproject_dir: 'modules',
)

ramsey_stabilizer_includes = ['include']

# submodules
qdac_client_proj = subproject('qdac-client')
shared_memory_handler_proj = subproject('shared-memory-handler')
labscript_address_utils_proj = subproject('labscript-address-utils')
hdf5 = subproject('hdf5-wrapper')
image_processing = subproject('image-processing')
configs_translator_proj = subproject('configs-translator')
feedback_controller_proj = subproject('feedback-controller')
awg_proj = subproject('awg')
microwave_awg_handler_proj = subproject('microwave-handler')

# dependencies
shared_memory_handler_dep = shared_memory_handler_proj.get_variable('shared_memory_handler_dep')
labscript_address_utils_dep = labscript_address_utils_proj.get_variable('labscript_address_utils_dep')
hdf5_dep = hdf5.get_variable('hdf5_wrapper_dep')
qdac_client_depe = qdac_client_proj.get_variable('qdac_client_dep')

opencv_dep = dependency('opencv', required: false)
if not opencv_dep.found()
    opencv_dep = dependency('opencv4', required: true)
endif

image_processing_dep = image_processing.get_variable('llrs_lib_image_processing_dep')
fftw_dep = dependency('fftw3', required: true)
configs_translator_dep = configs_translator_proj.get_variable('configs_translator_dep')
feedback_controller_dep = feedback_controller_proj.get_variable('feedback_controller_dep')
awg_dep = awg_proj.get_variable('awg_dep')
microwave_awg_handler_dep = microwave_awg_handler_proj.get_variable('microwave_awg_handler_dep')
nlopt_dep = dependency('nlopt', required: true)

ramsey_stabilizer_dependencies = [
    shared_memory_handler_dep, 
    labscript_address_utils_dep, 
    hdf5_dep, 
    opencv_dep, 
    image_processing_dep,
    fftw_dep,
    configs_translator_dep,
    feedback_controller_dep,
    awg_dep,
    microwave_awg_handler_dep,
    nlopt_dep,
    qdac_client_depe
]

# sources
ramsey_stabilizer_sources = [
    'src/ramsey-stabilizer.cu',
    'src/ramsey-stabilizer-metadata-saver.cu',
    'src/fourier-analyzer.cu',
]

# executable
ramsey_stabilizer_executable = executable(
    'ramsey-stabilizer',
    ['main.cu'] + ramsey_stabilizer_sources,
    link_args: '-lstdc++fs',
    dependencies: ramsey_stabilizer_dependencies,
    include_directories: ramsey_stabilizer_includes
)

ramsey_stabilizer_executable_test = executable(
    'ramsey-stabilizer-test',
    ['test.cu'] + ramsey_stabilizer_sources,
    link_args: '-lstdc++fs',
    dependencies: ramsey_stabilizer_dependencies,
    include_directories: ramsey_stabilizer_includes
)