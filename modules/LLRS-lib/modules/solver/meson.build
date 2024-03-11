project('llrs-lib-solver', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=static'])

# include directories
llrs_lib_solver_inc = include_directories('include')

# sources
llrs_lib_solver_src = [
    'src/Solver.cu',
  ]

# submodules
reconfiguration_proj =    subproject('reconfiguration')
collector_proj =          subproject('collector')

# dependencies
llrs_lib_reconfiguration_dep = reconfiguration_proj.get_variable('llrs_lib_reconfiguration_dep')
llrs_lib_collector_dep = collector_proj.get_variable('llrs_lib_collector_dep')

# targets
llrs_lib_solver = library('llrs-lib-solver', 
    llrs_lib_solver_src,
    include_directories: [llrs_lib_solver_inc],
    dependencies: [llrs_lib_reconfiguration_dep, llrs_lib_collector_dep])

# dependencies
llrs_lib_solver_dep = declare_dependency(
    include_directories: [llrs_lib_solver_inc],
    link_with: llrs_lib_solver,
    dependencies: [llrs_lib_reconfiguration_dep, llrs_lib_collector_dep])
