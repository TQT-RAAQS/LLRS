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
llcs_trigger_detector_proj = subproject('trigger-detector')
llcs_server_proj = subproject('server')
llcs_state_proj = subproject('state')
llcs_fsm_proj = subproject('fsm')

# dependencies

# external dependencies

# targets

# export dependencies
