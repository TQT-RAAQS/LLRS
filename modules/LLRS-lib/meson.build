project('LLRS-lib', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=shared'],
        subproject_dir: 'modules')


# common-include (proproc.h and settings.h to be included in all modules)
add_project_arguments('-I@CURRENT_SOURCE_DIR@/common/include', language: 'cuda')

# include directories
llrs_lib_inc = include_directories('include')

# sources
llrs_lib_src = [
    'src/llrs.cu'
  ]

# external dependencies
nlohmann_json_dep = dependency('nlohmann_json')
jsoncpp_dep = dependency('jsoncpp')

# submodules

collector_proj =          subproject('collector')
image_processing_proj =   subproject('image-processing')
jsonwrapper_proj =        subproject('jsonwrapper')
reconfiguration_proj =    subproject('reconfiguration')
setup_proj =              subproject('setup')
solver_proj =             subproject('solver')
utility_proj =            subproject('utility')
waveform_streaming_proj = subproject('waveform-streaming')
waveform_synthesis_proj = subproject('waveform-synthesis')

# targets
llrs_lib = shared_library('llrs-lib',
    llrs_lib_src,
    include_directories: [llrs_lib_inc],
    dependencies: [nlohmann_json_dep, jsoncpp_dep, llrs_lib_setup_dep, llrs_lib_jsonwrapper_dep, llrs_lib_collector_dep, llrs_lib_solver_dep, llrs_lib_waveform_synthesis_dep, llrs_lib_waveform_streaming_dep, llrs_lib_image_processing_dep, llrs_lib_utility_dep, fgc_dep, awg_dep],
    install: true)


