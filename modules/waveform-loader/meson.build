project('WaveformLoader', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=shared'],
        subproject_dir: 'modules')

# sources
waveform_loader_src = [
    'main.cu'
  ]

# submodules
setup_proj = subproject('setup')

# external dependencies
llrs_lib_setup_dep = setup_proj.get_variable('llrs_lib_setup_dep')

# targets
waveform_loader = executable('waveform-loader',
    waveform_loader_src,
    link_args: '-lstdc++fs',
    dependencies: [llrs_lib_setup_dep])
