project(
    'llcs-fsm',
    'cpp',
    default_options: ['cpp_std=c++14', 'default_library=static'],
)

# include directories
llcs_fsm_inc = include_directories('include')

# sources
llcs_fsm_src = ['src/fsm.cpp', 'src/llcs-config.cpp']

# submodules
state_proj = subproject('state')
llcs_server_handler_proj = subproject('llcs-server-handler')
trigger_detector_proj = subproject('trigger-detector')
hdf_wrapper_proj = subproject('hdf5-wrapper')
llrs_lib_proj = subproject('llrs-lib')

# dependencies
llcs_state_dep = state_proj.get_variable('llcs_state_dep')
llcs_server_handler_dep = llcs_server_handler_proj.get_variable(
    'llcs_server_handler_dep',
)
llcs_trigger_detector_dep = trigger_detector_proj.get_variable(
    'llcs_trigger_detector_dep',
)
hdf5_wrapper_dep = hdf_wrapper_proj.get_variable('hdf5_wrapper_dep')
llrs_lib_dep = llrs_lib_proj.get_variable('llrs_lib_dep')


# targets
llcs_fsm = library(
    'llcs_fsm',
    llcs_fsm_src,
    include_directories: [llcs_fsm_inc],
    dependencies: [
        llcs_state_dep,
        llcs_server_handler_dep,
        llcs_trigger_detector_dep,
        hdf5_wrapper_dep,
        llrs_lib_dep,
    ],
)

# dependencies
llcs_fsm_dep = declare_dependency(
    include_directories: [llcs_fsm_inc],
    link_with: [llcs_fsm],
    dependencies: [
        llcs_state_dep,
        llcs_server_handler_dep,
        llcs_trigger_detector_dep,
        hdf5_wrapper_dep,
        llrs_lib_dep,
    ],
)
