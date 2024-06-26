project('LLRS-lib', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=shared'],
        subproject_dir: 'modules')

# include directories
llrs_lib_inc = include_directories('include')

# sources
llrs_lib_src = [
    'src/llrs.cu'
  ]

# submodules
collector_proj =          subproject('collector')
image_processing_proj =   subproject('image-processing')
jsonwrapper_proj =        subproject('jsonwrapper')
setup_proj =              subproject('setup')
solver_proj =             subproject('solver')
utility_proj =            subproject('utility')
waveform_streaming_proj = subproject('waveform-streaming')
waveform_synthesis_proj = subproject('waveform-synthesis')
awg_proj =                subproject('awg')
fgc_proj =                subproject('fgc')

# dependencies
llrs_lib_collector_dep = collector_proj.get_variable('llrs_lib_collector_dep')
llrs_lib_image_processing_dep = image_processing_proj.get_variable('llrs_lib_image_processing_dep')
llrs_lib_jsonwrapper_dep = jsonwrapper_proj.get_variable('llrs_lib_jsonwrapper_dep')
llrs_lib_setup_dep = setup_proj.get_variable('llrs_lib_setup_dep')
llrs_lib_solver_dep = solver_proj.get_variable('llrs_lib_solver_dep')
llrs_lib_utility_dep = utility_proj.get_variable('llrs_lib_utility_dep')
llrs_lib_waveform_streaming_dep = waveform_streaming_proj.get_variable('llrs_lib_waveform_streaming_dep')
llrs_lib_waveform_synthesis_dep = waveform_synthesis_proj.get_variable('llrs_lib_waveform_synthesis_dep')
awg_dep = awg_proj.get_variable('awg_dep')
fgc_dep = fgc_proj.get_variable('fgc_dep')

# targets
llrs_lib = shared_library('llrs-lib',
    llrs_lib_src,
    link_args: '-lstdc++fs',
    include_directories: [llrs_lib_inc],
    dependencies: [llrs_lib_setup_dep, llrs_lib_jsonwrapper_dep, llrs_lib_collector_dep, llrs_lib_solver_dep, llrs_lib_waveform_synthesis_dep, llrs_lib_waveform_streaming_dep, llrs_lib_image_processing_dep, llrs_lib_utility_dep, fgc_dep, awg_dep],
    install: true)

llrs_lib_pre_solved = shared_library('llrs-lib-pre-solved',
    llrs_lib_src,
    cuda_args: ['-D LOGGING_RUNTIME', '-D PRE_SOLVED'],
    link_args: '-lstdc++fs',
    include_directories: [llrs_lib_inc],
    dependencies: [llrs_lib_setup_dep, llrs_lib_jsonwrapper_dep, llrs_lib_collector_dep, llrs_lib_solver_dep, llrs_lib_waveform_synthesis_dep, llrs_lib_waveform_streaming_dep, llrs_lib_image_processing_dep, llrs_lib_utility_dep, fgc_dep, awg_dep],
    install: true)

# export dependencies
llrs_lib_dep = declare_dependency(
    include_directories: llrs_lib_inc,
    link_with: llrs_lib,
    dependencies: [llrs_lib_setup_dep, llrs_lib_jsonwrapper_dep, llrs_lib_collector_dep, llrs_lib_solver_dep, llrs_lib_waveform_synthesis_dep, llrs_lib_waveform_streaming_dep, llrs_lib_image_processing_dep, llrs_lib_utility_dep, fgc_dep, awg_dep])

llrs_lib_pre_solved_dep = declare_dependency(
    include_directories: llrs_lib_inc,
    link_with: llrs_lib_pre_solved,
    dependencies: [llrs_lib_setup_dep, llrs_lib_jsonwrapper_dep, llrs_lib_collector_dep, llrs_lib_solver_dep, llrs_lib_waveform_synthesis_dep, llrs_lib_waveform_streaming_dep, llrs_lib_image_processing_dep, llrs_lib_utility_dep, fgc_dep, awg_dep])

