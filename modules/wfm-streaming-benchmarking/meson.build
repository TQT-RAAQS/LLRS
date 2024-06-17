project('WaveformStreamingBenchmark', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=shared'],
        subproject_dir: 'modules')

# sources
waveform_streaming_null_src = [
    'null.cu'
  ]

waveform_streaming_timing_src = [
    'timing.cu'
  ]

waveform_streaming_test_src = [
    'test.cu'
  ]

# submodules
awg_proj = subproject('awg')
waveform_synthesis_proj = subproject('waveform-synthesis')
waveform_streaming_proj = subproject('waveform-streaming')
solver_proj = subproject('solver')
setup_proj = subproject('setup')

# external dependencies
awg_dep = awg_proj.get_variable('awg_dep') 
waveform_synthesis_dep = waveform_synthesis_proj.get_variable('llrs_lib_waveform_synthesis_dep') 
waveform_streaming_dep = waveform_streaming_proj.get_variable('llrs_lib_waveform_streaming_dep') 
waveform_streaming_null_dep = waveform_streaming_proj.get_variable('llrs_lib_waveform_streaming_null_dep') 
solver_dep = solver_proj.get_variable('llrs_lib_solver_dep') 
setup_dep = setup_proj.get_variable('llrs_lib_setup_dep') 


# targets
null_test = executable('null-test',
    waveform_streaming_null_src,
    link_args: '-lstdc++fs',
    dependencies: [awg_dep, waveform_synthesis_dep, waveform_streaming_null_dep, solver_dep, setup_dep])

timing_test = executable('timing-test',
    waveform_streaming_timing_src,
    link_args: '-lstdc++fs',
    dependencies: [awg_dep, waveform_synthesis_dep, solver_dep, setup_dep])

waveform_streaming_test = executable('waveform-streaming-test',
    waveform_streaming_test_src,
    link_args: '-lstdc++fs',
    dependencies: [awg_dep, waveform_synthesis_dep])
