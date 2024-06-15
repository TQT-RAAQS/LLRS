project('WaveformStreamingTest', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=shared'],
        subproject_dir: 'modules')

# sources
waveform_streaming_test_src = [
    'main.cu'
  ]

# submodules
awg_proj = subproject('awg')
waveform_synthesis_proj = subproject('waveform-synthesis')

# dependencies
llrs_lib_waveform_synthesis_dep = waveform_synthesis_proj.get_variable('llrs_lib_waveform_synthesis_dep')
awg_dep = awg_proj.get_variable('awg_dep')

# targets
waveform_streaming_test = executable('waveform-streaming-test',
    waveform_streaming_test_src,
    link_args: '-lstdc++fs',
    dependencies: [awg_dep, llrs_lib_waveform_synthesis_dep])
