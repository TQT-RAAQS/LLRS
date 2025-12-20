project(
    'microwave-handler',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
    subproject_dir: 'modules',
)

cmake = import('cmake')
##########################

awg_proj = subproject('awg')
awg_dep = awg_proj.get_variable('awg_dep')

config_translator = subproject('configs-translator')
configs_translator_dep = config_translator.get_variable('configs_translator_dep')

timed_hash_queue = subproject('timed-hash-queue')
timed_hash_queue_dep = timed_hash_queue.get_variable('timed_hash_queue_dependency')

yaml_cpp = cmake.subproject('yaml-cpp')
yaml_cpp_dep = yaml_cpp.dependency('yaml-cpp')

dependencies = [
    awg_dep, 
    configs_translator_dep, 
    timed_hash_queue_dep, 
    yaml_cpp_dep, 
]

##########################

includes = ['include']
sources = [
    'src/microwave-awg-handler.cu',
    'src/microwave-waveforms.cu',
    'src/microwave-waveform-synthesizer.cu'
]

##########################

microwave_awg_handler_lib = library(
    'microwave-awg-handler-lib',
    sources,
    include_directories: includes,
    dependencies: dependencies,
)
microwave_awg_handler_dep = declare_dependency(
    include_directories: includes,
    link_with: microwave_awg_handler_lib,
    dependencies: dependencies
)

##########################

test_executable = executable(
    'microwave_test',
    ['test.cu'] + sources,
    link_args: '-lstdc++fs',
    dependencies: dependencies,
    include_directories: includes
)