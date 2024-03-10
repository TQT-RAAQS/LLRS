project('llrs-lib-setup', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=static'],
        subproject_dir: 'modules')


# include directories
llrs_lib_setup_inc = include_directories('include')

# sources
llrs_lib_setup_src = [
    'src/Setup.cu',
  ]


# targets
llrs_lib_setup = library('awg', 
    llrs_lib_setup_src,
    include_directories: [llrs_lib_common_inc, llrs_lib_setup_inc],
    dependencies: [waveform_synthesis_dep])

# dependencies
llrs_lib_setup_dep = declare_dependency(
    include_directories: [llrs_lib_common_inc, llrs_lib_setup_inc],
    link_with: llrs_lib_setup,
    dependencies: [waveform_synthesis_dep])
