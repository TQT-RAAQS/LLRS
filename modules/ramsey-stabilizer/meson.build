project(
    'ramsey-stabilizer',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
    subproject_dir: 'modules',
)

ramsey_stabilizer_includes = ['include']

# submodules
shared_memory_handler_proj = subproject('shared-memory-handler')
labscript_address_utils_proj = subproject('labscript-address-utils')
hdf5 = subproject('hdf5-wrapper')
image_processing = subproject('image-processing')
configs_translator_proj = subproject('configs-translator')
pid_loop_controller_proj = subproject('pid-loop-controller')
awg_proj = subproject('awg')

# dependencies
shared_memory_handler_dep = shared_memory_handler_proj.get_variable('shared_memory_handler_dep')
labscript_address_utils_dep = labscript_address_utils_proj.get_variable('labscript_address_utils_dep')
hdf5_dep = hdf5.get_variable('hdf5_wrapper_dep')
opencv_dep = dependency('opencv', required: true)
image_processing_dep = image_processing.get_variable('llrs_lib_image_processing_dep')
fftw_dep = dependency('fftw3', required: true)
configs_translator_dep = configs_translator_proj.get_variable('configs_translator_dep')
pid_loop_controller_dep = pid_loop_controller_proj.get_variable('pid_loop_controller_dep')
awg_dep = awg_proj.get_variable('awg_dep')

ramsey_stabilizer_dependencies = [
    shared_memory_handler_dep, 
    labscript_address_utils_dep, 
    hdf5_dep, 
    opencv_dep, 
    image_processing_dep,
    fftw_dep,
    configs_translator_dep,
    pid_loop_controller_dep,
    awg_dep
]

# sources
ramsey_stabilizer_sources = [
    'src/ramsey-stabilizer.cu',
    'src/ramsey-stabilizer-metadata-saver.cu',
]

ramsey_stabilizer_dep = []

# executable
ramsey_stabilizer_executable = executable(
    'ramsey-stabilizer',
    ['main.cu'] + ramsey_stabilizer_sources,
    link_args: '-lstdc++fs',
    dependencies: ramsey_stabilizer_dependencies,
    include_directories: ramsey_stabilizer_includes
)

ramsey_stabilizer_executable = executable(
    'ramsey-stabilizer-test',
    ['awg_test.cu'] + ramsey_stabilizer_sources,
    link_args: '-lstdc++fs',
    dependencies: ramsey_stabilizer_dependencies,
    include_directories: ramsey_stabilizer_includes
)