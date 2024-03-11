project('llrs-lib-waveform-streaming', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=static'])

# include directories
llrs_lib_waveform_streaming_inc = include_directories('include')

# sources
llrs_lib_solver_src = [
    'src/Sequence.cu',
  ]

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
