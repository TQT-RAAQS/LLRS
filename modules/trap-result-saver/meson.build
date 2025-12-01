project(
    'trap-result-saver',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
    subproject_dir: 'modules',
)

trap_result_saver_includes = ['include']

# submodules
shared_memory_handler_proj = subproject('shared-memory-handler')
labscript_address_utils_proj = subproject('labscript-address-utils')

# dependencies
shared_memory_handler_dep = shared_memory_handler_proj.get_variable('shared_memory_handler_dep')
labscript_address_utils_dep = labscript_address_utils_proj.get_variable('labscript_address_utils_dep')

trap_result_dependencies = [shared_memory_handler_dep, labscript_address_utils_dep]

# sources
trap_result_saver_sources = [
    'src/trap-result-saver.cu',
]

trap_result_saver_dep = []

# executable
trap_result_saver_executable = executable(
    'trap-result-saver',
    ['main.cu'] + trap_result_saver_sources,
    link_args: '-lstdc++fs',
    dependencies: trap_result_dependencies,
    include_directories: trap_result_saver_includes
)