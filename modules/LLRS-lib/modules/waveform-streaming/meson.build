project('llrs-lib-waveform-streaming', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=static'])

# include directories
llrs_lib_waveform_streaming_inc = include_directories('include')

# sources
llrs_lib_waveform_streaming_src = [
    'src/Sequence.cu',
  ]

# submodules
collector_proj =          subproject('collector')
setup_proj =              subproject('setup')
solver_proj =             subproject('solver')
waveform_synthesis_proj = subproject('waveform-synthesis')
awg_proj =                subproject('awg')

# dependencies
llrs_lib_collector_dep = collector_proj.get_variable('llrs_lib_collector_dep')
llrs_lib_setup_dep = setup_proj.get_variable('llrs_lib_setup_dep')
llrs_lib_solver_dep = solver_proj.get_variable('llrs_lib_solver_dep')
llrs_lib_waveform_synthesis_dep = waveform_synthesis_proj.get_variable('llrs_lib_waveform_synthesis_dep')
awg_dep = awg_proj.get_variable('awg_dep')

# targets
llrs_lib_waveform_streaming = library('llrs-lib-waveform-streaming', 
    llrs_lib_waveform_streaming_src,
    include_directories: [llrs_lib_waveform_streaming_inc],
    dependencies: [awg_dep, llrs_lib_collector_dep, llrs_lib_solver_dep, llrs_lib_waveform_synthesis_dep, llrs_lib_setup_dep])

# dependencies
llrs_lib_waveform_streaming_dep = declare_dependency(
    include_directories: [llrs_lib_waveform_streaming_inc],
    link_with: llrs_lib_waveform_streaming,
    dependencies: [awg_dep, llrs_lib_collector_dep, llrs_lib_solver_dep, llrs_lib_waveform_synthesis_dep, llrs_lib_setup_dep])
