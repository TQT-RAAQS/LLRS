project('llrs-lib-solver', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=static'])

# include directories
llrs_lib_solver_inc = include_directories('include')

# sources
llrs_lib_solver_src = [
    'src/Solver.cu',
  ]

# targets
llrs_lib_solver = library('llrs-lib-solver', 
    llrs_lib_solver_src,
    include_directories: [llrs_lib_common_inc, llrs_lib_solver_inc],
    dependencies: [llrs_lib_reconfiguration_dep, llrs_lib_collector_dep])

# dependencies
llrs_lib_solver_dep = declare_dependency(
    include_directories: [llrs_lib_common_inc, llrs_lib_solver_inc],
    link_with: llrs_lib_solver,
    dependencies: [])
