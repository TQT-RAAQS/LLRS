project('llcs-fsm', 
        'cpp', 
        default_options: ['cpp_std=c++14', 'default_library=static'])

# include directories
llcs_fsm_inc = include_directories('include')

# sources
llcs_fsm_src = [
    'src/fsm.cpp',
  ]

# submodules
state_proj =            subproject('state')
zmq_server_proj =           subproject('zmq-server')
trigger_detector_proj = subproject('trigger-detector')
h5_wrapper_proj =       subproject('h5-wrapper')
llrs_lib_proj =         subproject('llrs-lib')

# dependencies
llcs_state_dep = state_proj.get_variable('llcs_state_dep')
zmq_server_dep = zmq_server_proj.get_variable('zmq_server_dep')
llcs_trigger_detector_dep = trigger_detector_proj.get_variable('llcs_trigger_detector_dep')
h5_wrapper_dep = h5_wrapper_proj.get_variable('h5_wrapper_dep')
llrs_lib_dep = llrs_lib_proj.get_variable('llrs_lib_dep')


# targets
llcs_fsm = library('llcs_fsm', 
    llcs_fsm_src,
    include_directories: [llcs_fsm_inc],
    dependencies: [llcs_state_dep, zmq_server_dep, llcs_trigger_detector_dep, h5_wrapper_dep, llrs_lib_dep])

# dependencies
llcs_fsm_dep = declare_dependency(
    include_directories: [llcs_fsm_inc],
    link_with: [llcs_fsm],
    dependencies: [llcs_state_dep, zmq_server_dep, llcs_trigger_detector_dep, h5_wrapper_dep, llrs_lib_dep])
