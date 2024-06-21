project('LLCS', 
        'cpp', 
        default_options: ['cpp_std=c++14', 'default_library=shared'],
        subproject_dir: 'modules')

# sources
llcs_src = [
    'main.cpp'
  ]

# submodules
llcs_trigger_detector_proj = subproject('trigger-detector')
zmq_server_proj = subproject('zmq-server')
llcs_state_proj = subproject('state')
llcs_fsm_proj = subproject('fsm')
awg_proj = subproject('awg')

# external dependencies
llcs_trigger_detector_dep = llcs_trigger_detector_proj.get_variable('llcs_trigger_detector_dep')
zmq_server_dep = zmq_server_proj.get_variable('zmq_server_dep')
llcs_fsm_dep = llcs_fsm_proj.get_variable('llcs_fsm_dep')
awg_dep = awg_proj.get_variable('awg_dep')

# targets
llcs = executable('llcs',
    llcs_src,
    link_args: '-lstdc++fs',
    dependencies: [llcs_trigger_detector_dep, zmq_server_dep, llcs_fsm_dep, awg_dep],
    install: true)
