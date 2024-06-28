project(
    'llrs-lib-waveform-synthesis',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
)

cmake = import('cmake')

# include directories
llrs_lib_waveform_synthesis_inc = include_directories('include')

# external dependencies
yaml_cpp_proj = cmake.subproject('yaml-cpp')
yaml_cpp_dep = yaml_cpp_proj.dependency('yaml-cpp')

# sources
llrs_lib_waveform_src = ['src/Waveform.cu']

llrs_lib_waveform_power_src = ['src/WaveformPowerCalculator.cu']

llrs_lib_waveform_repo_src = ['src/WaveformRepo.cu']

llrs_lib_waveform_table_src = ['src/WaveformTable.cu']

# targets
llrs_lib_waveform = library(
    'llrs-lib-waveform',
    llrs_lib_waveform_src,
    include_directories: [llrs_lib_waveform_synthesis_inc],
    dependencies: [yaml_cpp_dep],
)

llrs_lib_waveform_power = library(
    'llrs-lib-waveform-power',
    llrs_lib_waveform_power_src,
    include_directories: [llrs_lib_waveform_synthesis_inc],
    dependencies: [yaml_cpp_dep],
)

llrs_lib_waveform_repo = library(
    'llrs-lib-waveform-repo',
    llrs_lib_waveform_repo_src,
    include_directories: [llrs_lib_waveform_synthesis_inc],
    link_with: [llrs_lib_waveform, llrs_lib_waveform_power],
    dependencies: [],
)

llrs_lib_waveform_table = library(
    'llrs-lib-waveform-table',
    llrs_lib_waveform_table_src,
    include_directories: [llrs_lib_waveform_synthesis_inc],
    link_with: llrs_lib_waveform_repo,
    dependencies: [],
)

# dependencies
llrs_lib_waveform_synthesis_dep = declare_dependency(
    include_directories: [llrs_lib_waveform_synthesis_inc],
    link_with: [llrs_lib_waveform_repo, llrs_lib_waveform_table],
    dependencies: [],
)
