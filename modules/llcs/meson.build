project('LLCS', 
        'cpp', 
        default_options: ['cpp_std=c++14', 'default_library=shared'],
        subproject_dir: 'modules')

# sources
llcs_src = [
    'main.cpp'
  ]

# submodules
h5_wrapper_proj = subproject('h5-wrapper')
trigger_detector_proj = subproject('trigger-detector')
llcs_server_proj = subproject('server')
llcs_state_proj = subproject('state')
llcs_fsm_proj = subproject('fsm')
awg_proj = subproject('awg')

# external dependencies
llcs_trigger_detector_dep = trigger_detector_proj.get_variable('llcs_trigger_detector_dep')
llcs_server_dep = llcs_server_proj.get_variable('llcs_server_dep')
llcs_fsm_dep = llcs_fsm_proj.get_variable('llcs_fsm_dep')
awg_dep = awg_proj.get_variable('awg_dep')

# targets
llcs = executable('llcs',
    llcs_src,
    link_args: '-lstdc++fs',
    dependencies: [llcs_trigger_detector_dep, llcs_server_dep, llcs_fsm_dep, awg_dep],
    install: true)
