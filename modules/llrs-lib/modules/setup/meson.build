project(
    'llrs-lib-setup',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
    subproject_dir: 'modules',
)

# include directories
llrs_lib_setup_inc = include_directories('include')

# sources
llrs_lib_setup_src = ['src/Setup.cu']

# submodules
waveform_synthesis_proj = subproject('waveform-synthesis')

# dependencies
llrs_lib_waveform_synthesis_dep = waveform_synthesis_proj.get_variable(
    'llrs_lib_waveform_synthesis_dep',
)

# targets
llrs_lib_setup = library(
    'llrs-lib-setup',
    llrs_lib_setup_src,
    include_directories: [llrs_lib_setup_inc],
    dependencies: [llrs_lib_waveform_synthesis_dep],
)

# dependencies
llrs_lib_setup_dep = declare_dependency(
    include_directories: [llrs_lib_setup_inc],
    link_with: llrs_lib_setup,
    dependencies: [llrs_lib_waveform_synthesis_dep],
)
