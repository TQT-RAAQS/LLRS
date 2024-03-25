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
waveform_synthesis_proj = subproject('waveform-synthesis')

# dependencies
llrs_lib_reconfiguration_dep = reconfiguration_proj.get_variable('llrs_lib_reconfiguration_dep')
llrs_lib_collector_dep = collector_proj.get_variable('llrs_lib_collector_dep')
llrs_lib_waveform_synthesis_dep = waveform_synthesis_proj.get_variable('llrs_lib_waveform_synthesis_dep')

# targets
llrs_lib_solver = library('llrs-lib-solver', 
    llrs_lib_solver_src,
    include_directories: [llrs_lib_solver_inc],
    dependencies: [llrs_lib_reconfiguration_dep, llrs_lib_collector_dep, llrs_lib_waveform_synthesis_dep])

llrs_lib_solver_so = shared_library('llrs-lib-solver-so',
    llrs_lib_solver_src,
    cuda_args: ['-D IS_SO'],
    include_directories: [llrs_lib_solver_inc],
    dependencies: [llrs_lib_reconfiguration_dep, llrs_lib_collector_dep, llrs_lib_waveform_synthesis_dep])    

# dependencies
llrs_lib_solver_dep = declare_dependency(
    include_directories: [llrs_lib_solver_inc],
    link_with: llrs_lib_solver,
    dependencies: [llrs_lib_reconfiguration_dep, llrs_lib_collector_dep, llrs_lib_waveform_synthesis_dep])
