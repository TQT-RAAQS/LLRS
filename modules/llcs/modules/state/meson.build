project('llcs-state', 
        'cpp', 
        default_options: ['cpp_std=c++14', 'default_library=static'])

# include directories
llcs_state_inc = include_directories('include')

# sources
llcs_state_src = [
    'src/state.cpp',
  ]

# targets
llcs_state = library('llrs-state', 
    llcs_state_src,
    include_directories: [llcs_state_inc],
    dependencies: [])

# dependencies
llcs_state_dep = declare_dependency(
    include_directories: [llcs_state_inc],
    link_with: llcs_state,
    dependencies: [])
